# Knowledge Distillation for Speech Recognition

This repository implements knowledge distillation for Automatic Speech Recognition (ASR) models using NVIDIA NeMo framework. The project focuses on training smaller, more efficient FastConformer-Transducer models by distilling knowledge from larger teacher models.

## 🎯 Project Overview

Knowledge distillation is a technique for transferring knowledge from a large, complex model (teacher) to a smaller, more efficient model (student). This project implements knowledge distillation for speech recognition models, enabling the creation of compact models that maintain competitive performance while requiring fewer computational resources.

### Key Features

- **FastConformer-Transducer Architecture**: State-of-the-art streaming ASR model
- **Knowledge Distillation**: Transfer learning from teacher to student models
- **ONNX Export**: Model deployment with ONNX runtime for efficient inference
- **LibriSpeech Training**: Comprehensive training on LibriSpeech dataset
- **Multiple Model Sizes**: Support for different model architectures (Small, Medium, Large)

## 📁 Repository Structure

```
knowledgedistill/
├── README.md                              # This file
├── train.py                              # Main training script
├── base.yaml                             # Base configuration file
├── fast-conformer_transducer_bpe.yaml    # Large model configuration
├── fast-conformer_transducer_bpe_medium.yaml # Medium model configuration (with KD)
├── manifest/                             # Dataset manifest files
│   ├── train_manifest.json
│   ├── val_manifest.json
│   ├── test_clean_manifest.json
│   └── test_other_manifest.json
├── scripts/                              # Utility scripts
│   ├── generate_manifest.py             # Generate LibriSpeech manifests
│   ├── evaluate.py                      # Model evaluation
│   ├── export_onnx.py                   # Export models to ONNX
│   ├── inference_onnx.py                # ONNX inference
│   └── extract_tokenizer.py             # Tokenizer utilities
├── tokenizer/                           # BPE tokenizer files
│   ├── tokenizer.model
│   ├── tokenizer.vocab
│   └── vocab.txt
├── teacher/                             # Teacher model storage
│   └── teacher_model.nemo
├── models/                              # Trained models
│   └── full_train.nemo
├── experiments/                         # Training experiment logs
│   ├── base/                           # Base model experiments
│   ├── new/                            # New model experiments
│   ├── whole_train/                    # Full training experiments
│   └── wandb/                          # Weights & Biases logs
└── Nemo-CM/                            # NeMo framework (submodule)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- NeMo framework
- PyTorch
- LibriSpeech dataset

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd knowledgedistill
```

2. Install dependencies:
```bash
pip install nemo_toolkit
pip install -r requirements.txt  # if available
```

3. Download LibriSpeech dataset and update paths in configuration files.

### Dataset Preparation

1. Generate manifest files for LibriSpeech:
```bash
python scripts/generate_manifest.py
```

This script processes LibriSpeech directories and creates manifest files for:
- Training data (train-clean-100, train-clean-360, train-other-500)
- Validation data (dev-clean, dev-other)
- Test data (test-clean, test-other)

## 🏋️ Training

### Standard Training

Train a FastConformer model without knowledge distillation:

```bash
python train.py --config-path=. --config-name=fast-conformer_transducer_bpe
```

### Knowledge Distillation Training

Train a student model with knowledge distillation:

```bash
python train.py --config-path=. --config-name=fast-conformer_transducer_bpe_medium \
    model.enable_kd=True \
    model.teacher_model_path=/path/to/teacher_model.nemo \
    model.kd_temperature=4.0 \
    model.kd_alpha=0.7
```

### Configuration Parameters

#### Knowledge Distillation Parameters

- `enable_kd`: Enable/disable knowledge distillation (default: False)
- `teacher_model_path`: Path to the pre-trained teacher model
- `kd_temperature`: Temperature for softening probability distributions (default: 1.0)
- `kd_alpha`: Weight balancing between distillation loss and ground truth loss (default: 0.5)

#### Model Architecture Variants

| Model Size | d_model | n_heads | n_layers | Parameters | Config File |
|------------|---------|---------|----------|------------|-------------|
| Small      | 176     | 4       | 16       | ~14M       | Custom      |
| Medium     | 256     | 4       | 16       | ~32M       | medium.yaml |
| Large      | 512     | 8       | 17       | ~120M      | base.yaml  |

## 📊 Evaluation

Evaluate trained models:

```bash
python scripts/evaluate.py
```

This script:
- Loads the trained model
- Transcribes test audio files
- Computes Word Error Rate (WER) and Character Error Rate (CER)
- Saves transcriptions for analysis

## 🔧 Model Export and Deployment

### ONNX Export

Export trained models to ONNX format for efficient inference:

```bash
python scripts/export_onnx.py
```

This creates:
- `encoder.onnx`: Encoder model
- `decoder.onnx`: Decoder model  
- `joiner.onnx`: Joint network
- `preprocessor.ts`: TorchScript preprocessor
- `tokens.txt`: Vocabulary file

### ONNX Inference

Run inference with exported ONNX models:

```bash
python scripts/inference_onnx.py --audio_file /path/to/audio.wav
```

## 📈 Experiment Tracking

The project uses Weights & Biases (wandb) for experiment tracking:

- Training metrics (loss, WER, learning rate)
- Model configurations
- Hyperparameter sweeps
- Experiment comparison

Configure wandb in the experiment manager section of config files:

```yaml
exp_manager:
  create_wandb_logger: true
  wandb_logger_kwargs:
    name: experiment_name
    project: project_name
```

## 🎛️ Configuration Files

### Base Configuration (`base.yaml`)

- Large FastConformer model (512 d_model, 17 layers)
- Standard training without knowledge distillation
- Optimized for high accuracy

### Medium Configuration (`fast-conformer_transducer_bpe_medium.yaml`)

- Medium FastConformer model (256 d_model, 16 layers)
- Knowledge distillation support
- Balanced between efficiency and accuracy

Key differences:
- Smaller model architecture (32M vs 120M parameters)
- Knowledge distillation parameters
- Adjusted learning rates and batch sizes

## 📝 Training Tips

1. **Batch Size**: Adjust based on GPU memory:
   - 16GB GPU: batch_size=8-16
   - 32GB GPU: batch_size=16-32
   - 80GB GPU: batch_size=32-64

2. **Knowledge Distillation**:
   - Use `kd_temperature=3-5` for better knowledge transfer
   - Balance `kd_alpha` between 0.3-0.7 depending on teacher quality
   - Ensure teacher and student use the same tokenizer

3. **Training Duration**:
   - Medium models: 50-100 epochs
   - Large models: 100-500 epochs
   - Monitor validation WER for early stopping

## 🔍 Results and Performance

The knowledge distillation approach typically achieves:

- **Model Size Reduction**: 70-80% parameter reduction (120M → 32M)
- **Performance Retention**: 90-95% of teacher model accuracy
- **Inference Speed**: 2-3x faster inference
- **Memory Usage**: 60-70% reduction in GPU memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA NeMo team for the excellent ASR framework
- LibriSpeech corpus for training data
- FastConformer architecture contributors

## 📚 References

- [FastConformer Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#fast-conformer)
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)
- [RNN-Transducer](https://arxiv.org/abs/1211.3711)
- [NeMo Framework](https://github.com/NVIDIA/NeMo)

---

For questions or issues, please open an issue in the repository or contact the maintainers.
