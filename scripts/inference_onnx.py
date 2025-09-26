#!/usr/bin/env python3
"""
Script to run ONNX inference with exported NeMo RNNT model.
This script uses ONNX models for encoder/decoder/joiner and TorchScript for preprocessor.
"""

import argparse
import numpy as np
import onnxruntime as ort
import torch
import librosa
import onnx
from typing import Dict, List, Tuple
import os


def load_onnx_metadata(model_path: str) -> Dict[str, str]:
    """Load metadata from ONNX model."""
    model = onnx.load(model_path)
    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value
    return metadata


def load_vocabulary(tokens_file: str) -> List[str]:
    """Load vocabulary from tokens file."""
    vocab = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = ' '.join(parts[:-1])  # Join all but last part (which is the index)
                vocab.append(token)
    return vocab


class ONNXRNNTInference:
    """ONNX RNNT inference class with streaming support."""
    
    def __init__(self, model_dir: str, tokens_file: str):
        """
        Initialize ONNX RNNT inference.
        
        Args:
            model_dir: Directory containing ONNX models (encoder.onnx, decoder.onnx, joiner.onnx) and TorchScript preprocessor (preprocessor.ts)
            tokens_file: Path to tokens.txt file
        """
        self.model_dir = model_dir
        self.tokens_file = tokens_file
        
        # Load models
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Load TorchScript preprocessor
        self.preprocessor_model = torch.jit.load(
            os.path.join(model_dir, 'preprocessor.ts'), map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.preprocessor_model.eval()
        self.encoder_session = ort.InferenceSession(
            os.path.join(model_dir, 'encoder.onnx'), providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            os.path.join(model_dir, 'decoder.onnx'), providers=providers
        )
        self.joiner_session = ort.InferenceSession(
            os.path.join(model_dir, 'joiner.onnx'), providers=providers
        )
        
        # Load metadata
        self.metadata = load_onnx_metadata(os.path.join(model_dir, 'encoder.onnx'))
        
        # Load vocabulary
        self.vocabulary = load_vocabulary(tokens_file)
        
        # Parse metadata
        self.sample_rate = int(self.metadata.get('sample_rate', 16000))
        self.chunk_shift = int(self.metadata.get('chunk_shift', 416))  # Streaming chunk shift
        self.vocab_size = int(self.metadata.get('vocab_size', len(self.vocabulary)))
        self.pred_rnn_layers = int(self.metadata.get('pred_rnn_layers', 2))
        self.pred_hidden = int(self.metadata.get('pred_hidden', 512))
        
        # Get preprocessor window parameters (these are in seconds, not samples)
        self.preprocessor_window_size = float(self.metadata.get('window_size', 0.025))
        self.preprocessor_window_stride = float(self.metadata.get('window_stride', 0.01))
        
        # Calculate streaming window size (chunk_shift + pre_encode_cache_size)
        # From export output: window_size = 425, chunk_shift = 416, so pre_encode_cache = 9
        self.window_size = self.chunk_shift + 9  # Approximate pre_encode_cache_size
        
        # Cache dimensions
        self.cache_last_channel_dim1 = int(self.metadata.get('cache_last_channel_dim1', 18))
        self.cache_last_channel_dim2 = int(self.metadata.get('cache_last_channel_dim2', 8))
        self.cache_last_channel_dim3 = int(self.metadata.get('cache_last_channel_dim3', 512))
        
        self.cache_last_time_dim1 = int(self.metadata.get('cache_last_time_dim1', 18))
        self.cache_last_time_dim2 = int(self.metadata.get('cache_last_time_dim2', 512))
        self.cache_last_time_dim3 = int(self.metadata.get('cache_last_time_dim3', 3))
        
        print(f"Loaded ONNX RNNT model with:")
        print(f"  Sample rate: {self.sample_rate}")
        print(f"  Streaming window size: {self.window_size}")
        print(f"  Chunk shift: {self.chunk_shift}")
        print(f"  Preprocessor window: {self.preprocessor_window_size}s")
        print(f"  Preprocessor stride: {self.preprocessor_window_stride}s")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Vocabulary: {self.vocabulary[:10]}...")  # Show first 10 tokens
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess audio file using the exported TorchScript preprocessor.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (processed_signal, processed_length)
        """
        # Load audio
        audio_signal, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Convert to tensor format expected by preprocessor
        audio_tensor = torch.from_numpy(audio_signal.astype(np.float32)).unsqueeze(0)  # Add batch dimension
        audio_length = torch.tensor([len(audio_signal)], dtype=torch.long)
        
        # Move to appropriate device (TorchScript models default to CPU, use CUDA if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        audio_tensor = audio_tensor.to(device)
        audio_length = audio_length.to(device)
        
        # Run preprocessor
        with torch.no_grad():
            processed_signal, processed_length = self.preprocessor_model(audio_tensor, audio_length)
        
        # Convert back to numpy for ONNX models
        processed_signal = processed_signal.cpu().numpy()
        processed_length = processed_length.cpu().numpy()
        
        return processed_signal, processed_length
    
    def initialize_decoder_state(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """Initialize decoder state for inference."""
        return {
            'hidden_state': np.zeros((1, batch_size, self.pred_hidden), dtype=np.float32),
            'cell_state': np.zeros((1, 1, self.pred_hidden), dtype=np.float32)
        }
    
    def run_encoder(self, processed_signal: np.ndarray, processed_length: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run encoder."""
        encoder_inputs = {
            self.encoder_session.get_inputs()[0].name: processed_signal,
            self.encoder_session.get_inputs()[1].name: processed_length
        }
        
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        
        return encoder_outputs[0], encoder_outputs[1]  # encoded, encoded_lengths
    
    def run_decoder_joiner(self, encoded: np.ndarray, encoded_lengths: np.ndarray) -> List[str]:
        """Run decoder and joiner for greedy decoding."""
        batch_size = encoded.shape[0]
        max_time = int(encoded_lengths[0])  # Use actual encoded length
        
        # Initialize decoder state
        hidden_state = np.zeros((1, batch_size, self.pred_hidden), dtype=np.float32)
        cell_state = np.zeros((1, 1, self.pred_hidden), dtype=np.float32)
        
        # Greedy decoding
        predictions = []
        
        # Start with blank token (vocab_size is blank token index)
        current_token = np.array([[self.vocab_size]], dtype=np.int32)  # Note: int32 for decoder
        
        for t in range(max_time):
            # Get encoder output at time t
            enc_out = encoded[:, :, t:t+1]  # (batch, 512, 1)
            
            # Run decoder with current token
            target_length = np.array([1], dtype=np.int32)
            decoder_inputs = {
                self.decoder_session.get_inputs()[0].name: current_token,  # targets
                self.decoder_session.get_inputs()[1].name: target_length,  # target_length
                self.decoder_session.get_inputs()[2].name: hidden_state,   # states.1
                self.decoder_session.get_inputs()[3].name: cell_state      # onnx::LSTM_3
            }
            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
            pred_out = decoder_outputs[0]  # (batch, 640, 1)
            hidden_state = decoder_outputs[2]  # Updated hidden state
            cell_state = decoder_outputs[3]    # Updated cell state
            
            # Run joiner
            joiner_inputs = {
                self.joiner_session.get_inputs()[0].name: enc_out,   # encoder_outputs
                self.joiner_session.get_inputs()[1].name: pred_out   # decoder_outputs
            }
            joiner_outputs = self.joiner_session.run(None, joiner_inputs)
            joint_out = joiner_outputs[0]  # (batch, 1, 1, vocab_size+1)
            
            # Get prediction (last dimension is vocab_size + blank)
            joint_out = joint_out.squeeze()  # Remove singleton dimensions -> (vocab_size+1,)
            pred_token_idx = np.argmax(joint_out)
            
            # If not blank token, add to predictions and update current token
            if pred_token_idx < self.vocab_size:  # Not blank token
                predictions.append(pred_token_idx)
                current_token = np.array([[pred_token_idx]], dtype=np.int32)
            else:
                # Blank token - don't add to predictions, keep current token
                pass
        
        # Convert predictions to text
        text = ''.join([self.vocabulary[idx] for idx in predictions if idx < len(self.vocabulary)])
        # Clean up SentencePiece tokens for better readability
        text = text.replace('▁', ' ').strip()
        return [text]
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Preprocess audio
        processed_signal, processed_length = self.preprocess_audio(audio_path)
        
        # Run encoder
        encoded, encoded_lengths = self.run_encoder(processed_signal, processed_length)
        
        # Run decoder and joiner
        transcriptions = self.run_decoder_joiner(encoded, encoded_lengths)
        
        return transcriptions[0] if transcriptions else ""


def main():
    parser = argparse.ArgumentParser(description='Run ONNX RNNT inference')
    parser.add_argument('--model_dir', type=str, default='/diyoData/experiments/knowledgedistill/model',
                       help='Directory containing ONNX models (default: current directory)')
    parser.add_argument('--tokens_file', type=str, default='/diyoData/experiments/knowledgedistill/model/tokens.txt',
                       help='Path to tokens.txt file (default: tokens.txt)')
    parser.add_argument('--audio_file', type=str, required=True,
                       help='Path to audio file to transcribe')
    
    args = parser.parse_args()
    
    # Check if model files exist
    model_files = ['encoder.onnx', 'decoder.onnx', 'joiner.onnx', 'preprocessor.ts']
    missing_files = []
    for file in model_files:
        if not os.path.exists(os.path.join(args.model_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing model files in {args.model_dir}: {missing_files}")
        print("Please run the export script first to generate the model files.")
        return
    
    if not os.path.exists(args.tokens_file):
        print(f"Error: Tokens file not found: {args.tokens_file}")
        return
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    print(f"Loading models from: {args.model_dir}")
    print(f"Using tokens file: {args.tokens_file}")
    
    # Initialize inference
    inference = ONNXRNNTInference(args.model_dir, args.tokens_file)
    
    # Transcribe audio
    print(f"Transcribing: {args.audio_file}")
    transcription = inference.transcribe(args.audio_file)
    print(f"Transcription: {transcription}")


if __name__ == "__main__":
    main()
