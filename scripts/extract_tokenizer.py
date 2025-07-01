import os
import shutil
import tempfile
import tarfile
import json
from nemo.collections.asr.models import EncDecRNNTBPEModel

nemo_path = "/diyoData/experiments/knowledgedistill/teacher/teacher_model.nemo"
tokenizer_dir = "tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

print(f"Loading model from {nemo_path}...")
model = EncDecRNNTBPEModel.restore_from(nemo_path)

# Method 1: Extract everything from the .nemo archive
print("\n=== Method 1: Extracting from .nemo archive ===")
try:
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting {nemo_path} to temporary directory...")
        with tarfile.open(nemo_path, 'r') as tar:
            tar.extractall(temp_dir)
            
        # List all files in the extracted archive
        print("Files in the archive:")
        all_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, temp_dir)
                all_files.append((rel_path, full_path))
                print(f"  {rel_path}")
        
        # Copy all potentially relevant files
        tokenizer_files_found = False
        for rel_path, full_path in all_files:
            if any(keyword in rel_path.lower() for keyword in ['tokenizer', '.model', 'vocab', 'spm']):
                dest_path = os.path.join(tokenizer_dir, os.path.basename(rel_path))
                shutil.copy2(full_path, dest_path)
                print(f"Copied: {rel_path} -> {dest_path}")
                tokenizer_files_found = True
        
        if not tokenizer_files_found:
            print("No obvious tokenizer files found in archive.")
            
except Exception as e:
    print(f"Archive extraction failed: {e}")

# Method 2: Inspect the tokenizer object more thoroughly
print("\n=== Method 2: Inspecting tokenizer object ===")
try:
    tokenizer = model.tokenizer
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Tokenizer attributes: {dir(tokenizer)}")
    
    if hasattr(tokenizer, 'tokenizer'):
        inner_tokenizer = tokenizer.tokenizer
        print(f"Inner tokenizer type: {type(inner_tokenizer)}")
        print(f"Inner tokenizer attributes: {dir(inner_tokenizer)}")
        
        # Try to access the SentencePiece processor directly
        if hasattr(inner_tokenizer, 'this'):
            sp_processor = inner_tokenizer.this
            print(f"SentencePiece processor type: {type(sp_processor)}")
            
            # Try different methods to get vocabulary
            print("\nTrying different vocabulary extraction methods:")
            
            # Method 2a: Try piece_to_id and id_to_piece with actual tokens
            try:
                print("Testing with common tokens...")
                test_tokens = ['<pad>', '<s>', '</s>', '<unk>', '▁', '▁the', '▁a', '▁and', 'a', 'e', 'i', 'o', 'u']
                for token in test_tokens:
                    try:
                        token_id = sp_processor.piece_to_id(token)
                        if token_id != sp_processor.unk_id():
                            back_to_piece = sp_processor.id_to_piece(token_id)
                            print(f"  {token} -> {token_id} -> {back_to_piece}")
                    except:
                        pass
            except Exception as e:
                print(f"Token testing failed: {e}")
            
            # Method 2b: Try to get the vocabulary size and sample some IDs
            try:
                vocab_size = inner_tokenizer.vocab_size
                print(f"Vocabulary size: {vocab_size}")
                
                # Sample some token IDs to see if we can get valid pieces
                sample_ids = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500]
                valid_pieces = {}
                
                for token_id in sample_ids:
                    if token_id < vocab_size:
                        try:
                            piece = sp_processor.id_to_piece(token_id)
                            if piece and not piece.startswith('<unk_'):
                                valid_pieces[token_id] = piece
                                print(f"  ID {token_id}: '{piece}'")
                        except:
                            pass
                
                if valid_pieces:
                    print(f"Found {len(valid_pieces)} valid pieces out of {len(sample_ids)} sampled")
                else:
                    print("No valid pieces found in sampling")
                    
            except Exception as e:
                print(f"Vocabulary sampling failed: {e}")
                
except Exception as e:
    print(f"Tokenizer inspection failed: {e}")

# Method 3: Try to use the model's actual tokenization methods
print("\n=== Method 3: Testing model tokenization ===")
try:
    test_texts = ["hello world", "the quick brown fox", "speech recognition", "नमस्ते", "测试"]
    
    for text in test_texts:
        try:
            # Use the model's tokenizer methods
            if hasattr(model.tokenizer, 'text_to_ids'):
                tokens = model.tokenizer.text_to_ids(text)
                decoded = model.tokenizer.ids_to_text(tokens)
                print(f"Text: '{text}' -> Tokens: {tokens} -> Decoded: '{decoded}'")
                
                # Try to get individual pieces
                if hasattr(model.tokenizer, 'tokenizer') and hasattr(model.tokenizer.tokenizer, 'this'):
                    sp = model.tokenizer.tokenizer.this
                    try:
                        pieces = sp.encode(text, out_type=str)
                        print(f"  Pieces: {pieces}")
                    except:
                        pass
            else:
                print(f"text_to_ids method not available")
                
        except Exception as e:
            print(f"Failed to tokenize '{text}': {e}")
            
except Exception as e:
    print(f"Model tokenization testing failed: {e}")

# Method 4: Try to access model configuration
print("\n=== Method 4: Checking model configuration ===")
try:
    if hasattr(model, 'cfg'):
        cfg = model.cfg
        print("Model configuration:")
        
        # Convert to dict if possible
        if hasattr(cfg, 'to_dict'):
            cfg_dict = cfg.to_dict()
        else:
            cfg_dict = dict(cfg) if hasattr(cfg, '__iter__') else str(cfg)
            
        print(json.dumps(cfg_dict, indent=2, default=str))
        
        # Save config
        config_file = os.path.join(tokenizer_dir, "model_config.json")
        with open(config_file, 'w') as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        print(f"Model config saved to: {config_file}")
        
except Exception as e:
    print(f"Config inspection failed: {e}")

# Method 5: Create a functional tokenizer wrapper
print("\n=== Method 5: Creating functional wrapper ===")
try:
    # Test that the tokenizer actually works
    test_text = "hello world"
    tokens = model.tokenizer.text_to_ids(test_text)
    decoded = model.tokenizer.ids_to_text(tokens)
    
    if decoded:
        print(f"Tokenizer is functional: '{test_text}' -> {tokens} -> '{decoded}'")
        
        # Create a more comprehensive wrapper
        wrapper_code = f'''#!/usr/bin/env python3
"""
Extracted tokenizer wrapper for NeMo model
Model: {nemo_path}
Vocab size: {model.tokenizer.tokenizer.vocab_size}
"""

import os
from nemo.collections.asr.models import EncDecRNNTBPEModel

class NeMoTokenizer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = "{nemo_path}"
        
        print(f"Loading NeMo model from {{model_path}}...")
        self.model = EncDecRNNTBPEModel.restore_from(model_path)
        self.vocab_size = self.model.tokenizer.tokenizer.vocab_size
        
    def encode(self, text):
        """Encode text to token IDs"""
        return self.model.tokenizer.text_to_ids(text)
    
    def decode(self, tokens):
        """Decode token IDs to text"""
        return self.model.tokenizer.ids_to_text(tokens)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.vocab_size
    
    def encode_as_pieces(self, text):
        """Encode text as pieces (if possible)"""
        try:
            sp = self.model.tokenizer.tokenizer.this
            return sp.encode(text, out_type=str)
        except:
            return None
    
    def __call__(self, text):
        """Make the tokenizer callable"""
        return self.encode(text)

# Example usage:
if __name__ == "__main__":
    tokenizer = NeMoTokenizer()
    
    test_texts = ["hello world", "speech recognition", "the quick brown fox"]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        pieces = tokenizer.encode_as_pieces(text)
        
        print(f"Text: '{{text}}'")
        print(f"  Tokens: {{tokens}}")
        print(f"  Decoded: '{{decoded}}'")
        if pieces:
            print(f"  Pieces: {{pieces}}")
        print()
'''
        
        wrapper_file = os.path.join(tokenizer_dir, "nemo_tokenizer.py")
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        print(f"Functional tokenizer wrapper saved to: {wrapper_file}")
        
        # Make it executable
        os.chmod(wrapper_file, 0o755)
        
    else:
        print("Tokenizer test failed - wrapper not created")
        
except Exception as e:
    print(f"Wrapper creation failed: {e}")

print(f"\n=== Summary ===")
print(f"Check the '{tokenizer_dir}' directory for extracted files:")
if os.path.exists(tokenizer_dir):
    for file in sorted(os.listdir(tokenizer_dir)):
        file_path = os.path.join(tokenizer_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size} bytes)")
else:
    print("  No files extracted")

print("\nRecommended next steps:")
print("1. Check if any .model files were extracted from the archive")
print("2. Use the nemo_tokenizer.py wrapper for tokenization")
print("3. If you need the raw SentencePiece model, try extracting it manually from the .nemo file")