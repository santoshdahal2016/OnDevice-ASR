import argparse
import os
from typing import Dict

from nemo.collections.asr.models import EncDecRNNTBPEModel
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

nemo_model_path = "/diyoData/experiments/knowledgedistill/experiments/full_train/2025-08-08_15-33-14/checkpoints/full_train.nemo"

def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


@torch.no_grad()
def main():
    # Create model directory if it doesn't exist
    model_dir = "/diyoData/experiments/knowledgedistill/model"
    os.makedirs(model_dir, exist_ok=True)

    asr_model = EncDecRNNTBPEModel.restore_from(nemo_model_path)

    # Save tokens.txt to model directory
    tokens_path = os.path.join(model_dir, "tokens.txt")
    with open(tokens_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print(f"Saved to {tokens_path}")

    asr_model.eval()

    assert asr_model.encoder.streaming_cfg is not None
    if isinstance(asr_model.encoder.streaming_cfg.chunk_size, list):
        chunk_size = asr_model.encoder.streaming_cfg.chunk_size[1]
    else:
        chunk_size = asr_model.encoder.streaming_cfg.chunk_size

    if isinstance(asr_model.encoder.streaming_cfg.pre_encode_cache_size, list):
        pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
    else:
        pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size
    window_size = chunk_size + pre_encode_cache_size

    print("chunk_size", chunk_size)
    print("pre_encode_cache_size", pre_encode_cache_size)
    print("window_size", window_size)

    chunk_shift = chunk_size

    # cache_last_channel: (batch_size, dim1, dim2, dim3)
    cache_last_channel_dim1 = len(asr_model.encoder.layers)
    cache_last_channel_dim2 = asr_model.encoder.streaming_cfg.last_channel_cache_size
    cache_last_channel_dim3 = asr_model.encoder.d_model

    # cache_last_time: (batch_size, dim1, dim2, dim3)
    cache_last_time_dim1 = len(asr_model.encoder.layers)
    cache_last_time_dim2 = asr_model.encoder.d_model
    cache_last_time_dim3 = asr_model.encoder.conv_context_size[0]

    # Export encoder, decoder, joiner to ONNX and preprocessor to TorchScript in model directory
    encoder_path = os.path.join(model_dir, "encoder.onnx")
    decoder_path = os.path.join(model_dir, "decoder.onnx")
    joiner_path = os.path.join(model_dir, "joiner.onnx")
    preprocessor_path = os.path.join(model_dir, "preprocessor.ts")
    
    asr_model.encoder.export(encoder_path)
    asr_model.decoder.export(decoder_path)
    asr_model.joint.export(joiner_path)
    # Export preprocessor to TorchScript since ONNX doesn't support STFT
    asr_model.preprocessor.export(preprocessor_path)
    print(f"Exported models to {model_dir}:")
    print(f"  - encoder.onnx")
    print(f"  - decoder.onnx") 
    print(f"  - joiner.onnx")
    print(f"  - preprocessor.ts")

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    # Get preprocessor configuration
    sample_rate = asr_model.cfg.preprocessor.sample_rate
    n_fft = asr_model.cfg.preprocessor.n_fft if hasattr(asr_model.cfg.preprocessor, 'n_fft') else None
    window_size = asr_model.cfg.preprocessor.window_size if hasattr(asr_model.cfg.preprocessor, 'window_size') else 0.02
    window_stride = asr_model.cfg.preprocessor.window_stride if hasattr(asr_model.cfg.preprocessor, 'window_stride') else 0.01
    n_mels = asr_model.cfg.preprocessor.features

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "window_size": window_size,
        "chunk_shift": chunk_shift,
        "normalize_type": normalize_type,
        "cache_last_channel_dim1": cache_last_channel_dim1,
        "cache_last_channel_dim2": cache_last_channel_dim2,
        "cache_last_channel_dim3": cache_last_channel_dim3,
        "cache_last_time_dim1": cache_last_time_dim1,
        "cache_last_time_dim2": cache_last_time_dim2,
        "cache_last_time_dim3": cache_last_time_dim3,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": f" ",
        "comment": "Encoder/decoder/joiner in ONNX format, preprocessor in TorchScript format",
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "window_size": window_size,
        "window_stride": window_stride,
        "n_mels": n_mels,
    }
    
    # Add metadata to all exported models
    add_meta_data(encoder_path, meta_data)
    add_meta_data(decoder_path, meta_data)
    add_meta_data(joiner_path, meta_data)
    print("Added metadata to all ONNX models")



if __name__ == "__main__":
    main()