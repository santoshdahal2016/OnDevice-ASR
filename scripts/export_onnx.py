

# from nemo.collections.asr.models import EncDecRNNTBPEModel


# nemo_model_path = "/diyoData/experiments/knowledgedistill/experiments/only_teacher/2025-07-09_20-46-05/checkpoints/only_teacher.nemo"

# model = EncDecRNNTBPEModel.restore_from(nemo_model_path)

# # Export ONNX model
# onnx_enc_model_fname = "onnx_export1" + "/" +  'encoder.onnx'
# onnx_dec_model_fname = "onnx_export1" + "/" +  'decoder.onnx'
# onnx_joint_model_fname = "onnx_export1" + "/" +  'joint.onnx'

# model.encoder.export(onnx_enc_model_fname)
# model.decoder.export(onnx_dec_model_fname)
# model.joint.export(onnx_joint_model_fname)


#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
import argparse
from typing import Dict

from nemo.collections.asr.models import EncDecRNNTBPEModel
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


nemo_model_path = "/diyoData/experiments/knowledgedistill/experiments/only_teacher/2025-07-09_20-46-05/checkpoints/only_teacher.nemo"



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

    asr_model = EncDecRNNTBPEModel.restore_from(nemo_model_path)

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

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


    asr_model.encoder.export("encoder.onnx")
    asr_model.decoder.export("decoder.onnx")
    asr_model.joint.export("joiner.onnx")

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

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
        "comment": "Only the transducer branch is exported",
    }
    add_meta_data("encoder.onnx", meta_data)

    for m in ["encoder", "decoder", "joiner"]:
        quantize_dynamic(
            model_input=f"{m}.onnx",
            model_output=f"{m}.int16.onnx",
            weight_type=QuantType.QUInt16,
        )

    print(meta_data)


if __name__ == "__main__":
    main()