"""T2S (GPT-SoVITS s1v3.ckpt) -> GGUF 转换器（路线 B 第一步）。

arch 名: gpt-sovits-t2s（自定义，llama.cpp 官方不识别；
本文件只保证 GGUF 容器合法 + 张量/超参完整，供自研 C++ runner 用）。

约定（ggml 惯例）：
- 2D Linear 权重 torch [out,in] -> 转置存 [in,out]
- embedding torch [vocab,dim] -> 转置存 [dim,vocab]（ggml_get_rows 用）
- bias/alpha/norm 按原样存
- 精度：保持 fp16（与管线一致）
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 仓库根（route_b/ 下脚本需回退两级）
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, "/home/swordreforge/llama.cpp/gguf-py")
from pathlib import Path
import torch
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

SRC = Path.home() / ".cache" / "gsv" / "s1v3.ckpt"
DST = Path.home() / ".cache" / "gsv" / "gguf" / "t2s-s1v3-f16.gguf"
ARCH = "gpt-sovits-t2s"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="微调 ckpt（同架构）-> gguf，同架构权重即插即用")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    print("loading ckpt...", flush=True)
    d = torch.load(src, map_location="cpu", weights_only=False)
    cfg = d["config"]["model"]
    w = d["weight"]
    print("tensors:", len(w), flush=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    wname = str(dst)
    writer = GGUFWriter(wname, ARCH)

    # hyper-parameters
    writer.add_uint32(f"{ARCH}.n_layer", cfg["n_layer"])
    writer.add_uint32(f"{ARCH}.hidden_dim", cfg["hidden_dim"])
    writer.add_uint32(f"{ARCH}.n_head", cfg["head"])
    writer.add_uint32(f"{ARCH}.vocab_size", cfg["vocab_size"])
    writer.add_uint32(f"{ARCH}.phoneme_vocab_size", cfg["phoneme_vocab_size"])
    writer.add_uint32(f"{ARCH}.eos_token_id", cfg["EOS"])
    writer.add_array(f"{ARCH}.suppressed_tokens", [280, 486, cfg["EOS"]])
    writer.add_string(f"{ARCH}.source_ckpt", src.name)

    def add(name, t, transpose=False):
        a = np.ascontiguousarray(t.detach().numpy())
        if transpose:
            a = np.ascontiguousarray(a.T)
        writer.add_tensor(name, a, raw_dtype=GGMLQuantizationType.F16)

    g = lambda n: w["model." + n]
    add(f"{ARCH}.ar_text_emb.weight", g("ar_text_embedding.word_embeddings.weight"), True)
    add(f"{ARCH}.ar_audio_emb.weight", g("ar_audio_embedding.word_embeddings.weight"), True)
    add(f"{ARCH}.ar_text_pos_alpha", g("ar_text_position.alpha"))
    add(f"{ARCH}.ar_audio_pos_alpha", g("ar_audio_position.alpha"))
    add(f"{ARCH}.bert_proj.weight", g("bert_proj.weight"), True)
    add(f"{ARCH}.bert_proj.bias", g("bert_proj.bias"))
    add(f"{ARCH}.predict.weight", g("ar_predict_layer.weight"), True)

    n_layer = cfg["n_layer"]
    for i in range(n_layer):
        p = f"h.layers.{i}."
        add(f"{ARCH}.blk.{i}.qkv.weight", w["model." + p + "self_attn.in_proj_weight"], True)
        add(f"{ARCH}.blk.{i}.qkv.bias", w["model." + p + "self_attn.in_proj_bias"])
        add(f"{ARCH}.blk.{i}.out.weight", w["model." + p + "self_attn.out_proj.weight"], True)
        add(f"{ARCH}.blk.{i}.out.bias", w["model." + p + "self_attn.out_proj.bias"])
        add(f"{ARCH}.blk.{i}.mlp1.weight", w["model." + p + "linear1.weight"], True)
        add(f"{ARCH}.blk.{i}.mlp1.bias", w["model." + p + "linear1.bias"])
        add(f"{ARCH}.blk.{i}.mlp2.weight", w["model." + p + "linear2.weight"], True)
        add(f"{ARCH}.blk.{i}.mlp2.bias", w["model." + p + "linear2.bias"])
        add(f"{ARCH}.blk.{i}.n1.weight", w["model." + p + "norm1.weight"])
        add(f"{ARCH}.blk.{i}.n1.bias", w["model." + p + "norm1.bias"])
        add(f"{ARCH}.blk.{i}.n2.weight", w["model." + p + "norm2.weight"])
        add(f"{ARCH}.blk.{i}.n2.bias", w["model." + p + "norm2.bias"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote:", wname, flush=True)


if __name__ == "__main__":
    main()
