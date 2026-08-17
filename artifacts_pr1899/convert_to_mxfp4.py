#!/usr/bin/env python3
"""Convert official Qwen block-FP8 shards to native MLX MXFP4 on disk."""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path("/Users/tim/Documents/talk/brainstorming/mlx-vlm-pr1899")
sys.path.insert(0, str(ROOT))

from mlx_vlm.models.qwen3_5.fp8 import _dequantize_qwen_fp8_weight  # noqa: E402

SRC = Path("/Users/tim/.cache/modelscope/hub/Qwen/Qwen3.8-27B-FP8")
DST = Path("/Users/tim/.cache/mlx-vlm/Qwen3.8-27B-MXFP4")
SKIP_WEIGHTS = {"mtp.safetensors"}
COPY_SUFFIXES = {".json", ".txt", ".jinja", ".py", ".model"}
COPY_NAMES = {"LICENSE", "README.md", "merges.txt", "vocab.json"}
MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}


def copy_sidecars() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for item in SRC.iterdir():
        if item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.suffix in COPY_SUFFIXES or item.name in COPY_NAMES:
            shutil.copy2(item, DST / item.name)


def quantize_one(weight: mx.array, scale_inv: mx.array):
    if weight.shape[-1] % MXFP4["group_size"] != 0:
        raise ValueError(f"incompatible shape {weight.shape}")
    restored = _dequantize_qwen_fp8_weight(weight, scale_inv)
    quantized = mx.quantize(restored, **MXFP4)
    packed, scales = quantized[0], quantized[1]
    mx.eval(packed, scales)
    mx.synchronize()
    del restored
    mx.clear_cache()
    return packed, scales


def convert_shard(src: Path, dst: Path) -> dict[str, str]:
    t0 = time.time()
    weights = mx.load(str(src))
    scale_keys = [key for key in weights if key.endswith(".weight_scale_inv")]
    print(
        f"[convert] {src.name} tensors={len(weights)} fp8_pairs={len(scale_keys)}",
        flush=True,
    )
    for i, scale_key in enumerate(scale_keys, 1):
        weight_key = scale_key[: -len("_scale_inv")]
        weight = weights.pop(weight_key)
        scale_inv = weights.pop(scale_key)
        packed, scales = quantize_one(weight, scale_inv)
        del weight, scale_inv
        weights[weight_key] = packed
        weights[weight_key[: -len(".weight")] + ".scales"] = scales
        if i == 1 or i == len(scale_keys) or i % 25 == 0:
            print(f"[convert] {src.name} {i}/{len(scale_keys)} {weight_key}", flush=True)
            gc.collect()
    leftover = [key for key in weights if key.endswith(".weight_scale_inv")]
    if leftover:
        raise RuntimeError(f"{src.name} still has scale_inv: {leftover[:3]}")
    mx.save_safetensors(str(dst), weights, metadata={"format": "mlx"})
    mapping = {key: dst.name for key in weights}
    del weights
    mx.clear_cache()
    gc.collect()
    print(f"[convert] wrote {dst.name} in {time.time() - t0:.1f}s", flush=True)
    return mapping


def patch_config() -> None:
    cfg_path = DST / "config.json"
    cfg = json.loads(cfg_path.read_text())
    quant = dict(MXFP4)
    cfg["quantization"] = quant
    cfg["quantization_config"] = quant
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        text_cfg.pop("quantization_config", None)
    cfg.pop("torch_dtype", None)
    cfg_path.write_text(json.dumps(dict(sorted(cfg.items())), indent=4) + "\n")


def write_index(weight_map: dict[str, str], total_size: int) -> None:
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (DST / "model.safetensors.index.json").write_text(json.dumps(index, indent=4) + "\n")


def main() -> None:
    print(f"[convert] src={SRC}", flush=True)
    print(f"[convert] dst={DST}", flush=True)
    if DST.exists():
        shutil.rmtree(DST)
    copy_sidecars()
    shards = sorted(p for p in SRC.glob("*.safetensors") if p.name not in SKIP_WEIGHTS)
    print(f"[convert] shards={len(shards)}", flush=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, src in enumerate(shards, 1):
        print(f"[convert] shard {i}/{len(shards)} {src.name}", flush=True)
        mapping = convert_shard(src, DST / src.name)
        weight_map.update(mapping)
        total_size += (DST / src.name).stat().st_size
    leftover = [key for key in weight_map if key.endswith(".weight_scale_inv")]
    scales = sum(1 for key in weight_map if key.endswith(".scales"))
    print(
        f"[convert] done keys={len(weight_map)} scales={scales} leftover_fp8={len(leftover)} bytes={total_size}",
        flush=True,
    )
    if leftover:
        raise SystemExit("conversion left weight_scale_inv on disk")
    write_index(weight_map, total_size)
    patch_config()
    print("[convert] ready", flush=True)


if __name__ == "__main__":
    main()
