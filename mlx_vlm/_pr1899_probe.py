"""Local-only load/convert probes for reviewing mlx-vlm PR 1899.

Does not change conversion math. Writes JSONL plus stderr heartbeats.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

LOG_PATH = Path(
    os.environ.get(
        "PR1899_PROBE_LOG",
        str(Path(__file__).resolve().parents[1] / "artifacts_pr1899" / "load-probe.jsonl"),
    )
)


def _metal_bytes() -> dict:
    try:
        import mlx.core as mx

        out = {}
        if hasattr(mx, "get_active_memory"):
            out["active_bytes"] = int(mx.get_active_memory())
        if hasattr(mx, "get_peak_memory"):
            out["peak_bytes"] = int(mx.get_peak_memory())
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            out["metal_active_bytes"] = int(mx.metal.get_active_memory())
            out["metal_peak_bytes"] = int(mx.metal.get_peak_memory())
        return out
    except Exception as exc:
        return {"metal_error": repr(exc)}


def _emit(event: str, **fields) -> None:
    rec = {
        "ts": time.time(),
        "event": event,
        **_metal_bytes(),
        **fields,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
    extra = " ".join(f"{k}={v}" for k, v in fields.items() if k in {
        "i", "n", "key", "shape", "in_dtype", "out_dtype", "elapsed_s",
        "supported", "model", "phase", "msg",
    })
    print(f"[pr1899-probe] {event} {extra}", flush=True)


def install() -> None:
    from mlx_vlm.models.qwen3_5 import fp8 as fp8_mod
    import mlx_vlm as pkg

    if getattr(fp8_mod.convert_qwen_fp8_weights, "_pr1899_probed", False):
        return

    orig_make = fp8_mod.make_quantization_config
    orig_convert = fp8_mod.convert_qwen_fp8_weights
    orig_quant_one = fp8_mod.quantize_qwen_fp8_weight
    orig_load = pkg.load

    def make_quantization_config(config: dict):
        t0 = time.time()
        qcfg = config.get("quantization_config") or {}
        result = orig_make(config)
        _emit(
            "make_quantization_config",
            model_type=config.get("model_type"),
            quant_method=qcfg.get("quant_method") if isinstance(qcfg, dict) else type(qcfg).__name__,
            fmt=qcfg.get("fmt") if isinstance(qcfg, dict) else None,
            weight_block_size=qcfg.get("weight_block_size") if isinstance(qcfg, dict) else None,
            result=result,
            elapsed_s=round(time.time() - t0, 4),
        )
        return result

    _quant_i = {"n": 0}

    def quantize_qwen_fp8_weight(weight, scale_inv):
        _quant_i["n"] += 1
        t0 = time.time()
        packed, scales = orig_quant_one(weight, scale_inv)
        elapsed = time.time() - t0
        i = _quant_i["n"]
        if i == 1 or i % 25 == 0:
            _emit(
                "convert_tensor",
                i=i,
                key="?",
                shape=tuple(weight.shape),
                in_dtype=str(weight.dtype),
                out_dtype=str(packed.dtype),
                elapsed_s=round(elapsed, 4),
            )
        return packed, scales

    def convert_qwen_fp8_weights(weights: dict):
        n = sum(1 for key in weights if key.endswith(".weight_scale_inv"))
        _quant_i["n"] = 0
        _emit("convert_start", n=n, n_tensors=len(weights))
        t_all = time.time()
        out = orig_convert(weights)
        _emit(
            "convert_done",
            n=n,
            n_out=len(out),
            elapsed_s=round(time.time() - t_all, 3),
        )
        return out

    def load(model_path, *args, **kwargs):
        _emit("load_start", model=str(model_path), kwargs=sorted(kwargs))
        t0 = time.time()
        try:
            out = orig_load(model_path, *args, **kwargs)
            _emit("load_done", model=str(model_path), elapsed_s=round(time.time() - t0, 3))
            return out
        except Exception:
            _emit(
                "load_fail",
                model=str(model_path),
                elapsed_s=round(time.time() - t0, 3),
                error=traceback.format_exc()[-2000:],
            )
            raise

    fp8_mod.make_quantization_config = make_quantization_config
    fp8_mod.quantize_qwen_fp8_weight = quantize_qwen_fp8_weight
    fp8_mod.convert_qwen_fp8_weights = convert_qwen_fp8_weights
    from mlx_vlm.models.qwen3_5 import qwen3_5 as qwen3_5_mod

    qwen3_5_mod.convert_qwen_fp8_weights = convert_qwen_fp8_weights
    try:
        from mlx_vlm.speculative.drafters.qwen3_5_mtp import qwen3_5_mtp as mtp_mod

        mtp_mod.convert_qwen_fp8_weights = convert_qwen_fp8_weights
    except Exception as exc:
        _emit("mtp_patch_skip", error=repr(exc))
    pkg.load = load
    convert_qwen_fp8_weights._pr1899_probed = True
    _emit("probe_installed", log=str(LOG_PATH))
    print(f"[pr1899-probe] log={LOG_PATH}", flush=True)
