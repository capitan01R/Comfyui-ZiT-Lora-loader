#!/usr/bin/env python3
"""
lora_meta.py — Deep forensic analysis of .safetensors LoRA files.
Reverse-engineers rank, alpha, architecture, layer coverage, weight stats,
training signal strength, and more directly from tensor data.
No args. No deps beyond stdlib + numpy (falls back gracefully without it).
"""

import json
import struct
import sys
import math
import os
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ─────────────────────────────────────────────────────────────────────────────
# SAFETENSORS READER
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

def read_header(path: Path):
    with open(path, "rb") as f:
        raw_len = f.read(8)
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_header = f.read(header_len)
        data_offset = 8 + header_len
    header = json.loads(raw_header.decode("utf-8"))
    meta = header.pop("__metadata__", {})
    return meta, header, data_offset

def read_tensor_bytes(path: Path, info: dict, data_offset: int) -> bytes:
    dtype = info.get("dtype", "F32")
    offsets = info.get("data_offsets", [0, 0])
    start, end = offsets[0], offsets[1]
    with open(path, "rb") as f:
        f.seek(data_offset + start)
        return f.read(end - start)

def bytes_to_floats(raw: bytes, dtype: str):
    if not HAS_NUMPY:
        return None
    dt_map = {
        "F32": np.float32, "F64": np.float64,
        "F16": np.float16, "BF16": "bfloat16",
        "I32": np.int32,   "I16": np.int16, "I8": np.int8,
    }
    if dtype == "BF16":
        # reinterpret as uint16, shift to float32
        arr16 = np.frombuffer(raw, dtype=np.uint16)
        arr32 = arr16.astype(np.uint32) << 16
        return arr32.view(np.float32)
    dt = dt_map.get(dtype)
    if dt is None:
        return None
    return np.frombuffer(raw, dtype=dt)

def parse_json_field(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            pass
    return v

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_architecture(keys):
    k = "\n".join(keys).lower()
    if "double_blocks" in k or "single_blocks" in k:
        if "klein" in k or "flux2" in k or "9b" in k:
            return "FLUX.2 Klein"
        return "FLUX.1 / FLUX.2"
    if "transformer_blocks" in k and "unet" not in k:
        return "FLUX-like DiT"
    if "up_blocks" in k and "text_model" in k:
        return "SDXL"
    if "up_blocks" in k:
        return "SD 1.x/2.x"
    if "mmdit" in k:
        return "SD3"
    return "Unknown"

def detect_lora_type(keys):
    k = "\n".join(keys).lower()
    if "dora_scale" in k or "magnitude" in k:
        return "DoRA (weight-decomposed)"
    if "lokr" in k or "kron" in k:
        return "LoKr"
    if "loha" in k or "hada" in k:
        return "LoHa"
    if "lora_down" in k or "lora_a" in k:
        return "Standard LoRA"
    return "Unknown"

# ─────────────────────────────────────────────────────────────────────────────
# LAYER PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_lora_key(key: str):
    """
    Returns (base_key, role) where role is one of:
      lora_down / lora_up / alpha / dora_scale / bias / other
    """
    k = key
    for suffix in ["lora_down.weight", "lora_up.weight",
                   "lora_A.weight", "lora_B.weight",
                   "alpha", "dora_scale", "bias"]:
        if k.endswith("." + suffix) or k.endswith("_" + suffix):
            base = k[: -(len(suffix) + 1)]
            role = suffix.replace(".weight", "").replace("lora_A", "lora_down").replace("lora_B", "lora_up")
            return base, role
    return k, "other"

def layer_type(base_key: str):
    k = base_key.lower()
    if any(x in k for x in ["to_q", "attn_q", "q_proj"]):   return "attn_q"
    if any(x in k for x in ["to_k", "attn_k", "k_proj"]):   return "attn_k"
    if any(x in k for x in ["to_v", "attn_v", "v_proj"]):   return "attn_v"
    if any(x in k for x in ["to_out", "proj_out", "out_proj"]): return "attn_out"
    if any(x in k for x in ["ff", "mlp", "feed_forward"]):  return "ff/mlp"
    if "proj" in k:                                           return "proj"
    if "norm" in k:                                           return "norm"
    if "embed" in k:                                          return "embed"
    return "other"

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT STATS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(arr):
    if arr is None or len(arr) == 0:
        return {}
    arr = arr.astype(np.float32) if HAS_NUMPY else arr
    finite = arr[np.isfinite(arr)] if HAS_NUMPY else arr
    if len(finite) == 0:
        return {"all_nan_or_inf": True}
    return {
        "mean":   float(np.mean(finite)),
        "std":    float(np.std(finite)),
        "min":    float(np.min(finite)),
        "max":    float(np.max(finite)),
        "l2":     float(np.sqrt(np.sum(finite ** 2))),
        "l1":     float(np.mean(np.abs(finite))),
        "nz_pct": float(100.0 * np.count_nonzero(finite) / len(finite)),
    }

def effective_rank(arr, shape):
    """Estimate effective rank via SVD on the reshaped down weight matrix."""
    if not HAS_NUMPY or arr is None:
        return None
    try:
        mat = arr.reshape(shape[0], -1).astype(np.float32)
        s = np.linalg.svd(mat, compute_uv=False)
        s = s[s > 1e-6]
        if len(s) == 0:
            return 0
        s_norm = s / s.sum()
        entropy = -float(np.sum(s_norm * np.log(s_norm + 1e-12)))
        return round(math.exp(entropy), 2)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# PRETTY OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

W = 72

def section(t):
    print(f"\n{'═' * W}\n  {t}\n{'═' * W}")

def sub(t):
    pad = W - 6 - len(t)
    print(f"\n  ── {t} {'─' * max(0, pad)}")

def row(label, value, w=36, indent=4):
    s = str(value)
    if len(s) > 100: s = s[:97] + "..."
    print(f"{' '*indent}{label:<{w}} {s}")

def table_header(*cols, widths):
    header = "  " + "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("  " + "─" * (sum(widths)))

def table_row(*cols, widths):
    print("  " + "".join(f"{str(c):<{w}}" for c, w in zip(cols, widths)))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse(path: Path):
    section(f"FORENSIC ANALYSIS: {path.name}")
    row("Path",  path.resolve())
    row("Size",  f"{path.stat().st_size / 1024 / 1024:.2f} MB")

    meta, header, data_offset = read_header(path)

    # ── Metadata dump ─────────────────────────────────────────────────────
    sub("Embedded Metadata")
    if meta:
        for k in sorted(meta.keys()):
            v = parse_json_field(meta[k])
            if isinstance(v, (dict, list)):
                print(f"    {k}:")
                for line in json.dumps(v, indent=6).splitlines():
                    print(f"      {line}")
            else:
                row(k, v)
    else:
        print("    (none)")

    # ── Inventory all tensor keys ─────────────────────────────────────────
    all_keys = list(header.keys())
    arch = detect_architecture(all_keys)
    lora_type = detect_lora_type(all_keys)

    sub("Architecture Fingerprint")
    row("Architecture",  arch)
    row("LoRA type",     lora_type)
    row("Total tensors", len(all_keys))

    # ── Parse into (base_key → {role: tensor_info}) ───────────────────────
    layers = defaultdict(dict)
    for key, info in header.items():
        base, role = parse_lora_key(key)
        layers[base][role] = info

    # ── Rank analysis from shapes ─────────────────────────────────────────
    sub("Rank Analysis (from tensor shapes)")
    ranks = defaultdict(int)
    alpha_values = {}
    layer_ranks = {}

    for base, roles in layers.items():
        down = roles.get("lora_down") or roles.get("lora_A")
        up   = roles.get("lora_up")   or roles.get("lora_B")
        alp  = roles.get("alpha")

        if down:
            shape = down.get("shape", [])
            if shape:
                r = shape[0]  # rank is first dim of down weight
                ranks[r] += 1
                layer_ranks[base] = r

        if alp:
            # alpha is stored as a scalar tensor — read it
            try:
                raw = read_tensor_bytes(path, alp, data_offset)
                arr = bytes_to_floats(raw, alp.get("dtype", "F32"))
                if arr is not None and len(arr) > 0:
                    alpha_values[base] = float(arr[0])
            except Exception:
                pass

    if ranks:
        print(f"    {'Rank':<10} {'Layer count':>12}")
        print(f"    {'─'*10} {'─'*12}")
        for r, cnt in sorted(ranks.items()):
            print(f"    {r:<10} {cnt:>12}")
    else:
        print("    (could not determine rank from shapes)")

    if alpha_values:
        unique_alphas = sorted(set(round(v, 4) for v in alpha_values.values()))
        row("\nAlpha values found", unique_alphas)
        # infer scale = alpha / rank
        for base, a in list(alpha_values.items())[:3]:
            r = layer_ranks.get(base)
            if r:
                row(f"  scale ({base[:40]})", f"{a}/{r} = {a/r:.4f}")

    # ── Layer coverage breakdown ──────────────────────────────────────────
    sub("Layer Coverage")
    type_counts = defaultdict(int)
    for base in layers:
        type_counts[layer_type(base)] += 1

    table_header("Layer type", "Count", widths=[30, 10])
    for lt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        table_row(lt, cnt, widths=[30, 10])

    # ── Per-layer shape table ─────────────────────────────────────────────
    sub("Per-Layer Shape Table (down / up / rank)")
    print(f"    {'Layer (truncated)':<52} {'down shape':<20} {'up shape':<20} {'rank':>6}")
    print(f"    {'─'*52} {'─'*20} {'─'*20} {'─'*6}")

    for base in sorted(layers.keys()):
        roles = layers[base]
        dn = roles.get("lora_down") or roles.get("lora_A")
        up = roles.get("lora_up")   or roles.get("lora_B")
        dn_shape = str(dn["shape"]) if dn and "shape" in dn else "—"
        up_shape = str(up["shape"]) if up and "shape" in up else "—"
        rank     = str(layer_ranks.get(base, "?"))
        label    = base[-52:] if len(base) > 52 else base
        print(f"    {label:<52} {dn_shape:<20} {up_shape:<20} {rank:>6}")

    # ── Weight statistics (needs numpy) ──────────────────────────────────
    if HAS_NUMPY:
        sub("Weight Statistics per Layer (requires numpy)")
        print(f"\n    Analysing {len(layers)} layers — this may take a moment...\n")

        grand_norms = []
        layer_stats = []

        for base in sorted(layers.keys()):
            roles = layers[base]
            dn_info = roles.get("lora_down") or roles.get("lora_A")
            up_info = roles.get("lora_up")   or roles.get("lora_B")

            if not dn_info or not up_info:
                continue

            try:
                dn_raw = read_tensor_bytes(path, dn_info, data_offset)
                up_raw = read_tensor_bytes(path, up_info, data_offset)
                dn_arr = bytes_to_floats(dn_raw, dn_info.get("dtype", "F32"))
                up_arr = bytes_to_floats(up_raw, up_info.get("dtype", "F32"))
            except Exception as e:
                layer_stats.append((base, None, None, str(e)))
                continue

            dn_stats = compute_stats(dn_arr)
            up_stats = compute_stats(up_arr)

            # Effective product weight norm: approximate ΔW = up @ down
            try:
                dn_shape = dn_info["shape"]
                up_shape = up_info["shape"]
                dn_mat = dn_arr.reshape(dn_shape[0], -1).astype(np.float32)
                up_mat = up_arr.reshape(up_shape[0], -1).astype(np.float32)
                # ΔW = up_mat @ dn_mat  (out x in)
                delta_w = up_mat @ dn_mat
                dw_norm   = float(np.linalg.norm(delta_w, 'fro'))
                dw_max    = float(np.max(np.abs(delta_w)))
                eff_rank  = effective_rank(dn_arr, dn_shape)
                alpha     = alpha_values.get(base)
                rank_val  = layer_ranks.get(base, 1)
                scale     = (alpha / rank_val) if alpha else 1.0
                dw_scaled = dw_norm * scale
                grand_norms.append(dw_scaled)
                layer_stats.append((base, dn_stats, up_stats, {
                    "dw_frob_norm":   round(dw_norm,   4),
                    "dw_scaled_norm": round(dw_scaled, 4),
                    "dw_max_weight":  round(dw_max,    6),
                    "effective_rank": eff_rank,
                    "alpha":          round(alpha, 4) if alpha else "n/a",
                    "scale":          round(scale, 4),
                }))
            except Exception as e:
                layer_stats.append((base, dn_stats, up_stats, f"(delta-W error: {e})"))

        # Print table
        print(f"    {'Layer':<48} {'ΔW scaled‑norm':>16} {'eff‑rank':>10} {'α/rank':>8} {'↑ signal?':>10}")
        print(f"    {'─'*48} {'─'*16} {'─'*10} {'─'*8} {'─'*10}")

        # Compute median norm for signal flag
        if grand_norms:
            median_norm = sorted(grand_norms)[len(grand_norms)//2]
        else:
            median_norm = 1.0

        for base, dn_s, up_s, extra in layer_stats:
            label = base[-48:] if len(base) > 48 else base
            if isinstance(extra, dict):
                n   = f"{extra['dw_scaled_norm']:.4f}"
                er  = str(extra['effective_rank']) if extra['effective_rank'] else "n/a"
                sc  = str(extra['scale'])
                sig = "  ★" if extra['dw_scaled_norm'] > median_norm * 1.5 else ""
                print(f"    {label:<48} {n:>16} {er:>10} {sc:>8} {sig:>10}")
            else:
                print(f"    {label:<48} {'error':>16} {'—':>10} {'—':>8} {'':>10}")

        # Summary
        sub("Training Signal Summary")
        if grand_norms:
            row("Layers with weight stats",  len(grand_norms))
            row("Mean scaled ΔW norm",        round(float(np.mean(grand_norms)), 4))
            row("Median scaled ΔW norm",      round(float(np.median(grand_norms)), 4))
            row("Max scaled ΔW norm",         round(float(np.max(grand_norms)), 4))
            row("Min scaled ΔW norm",         round(float(np.min(grand_norms)), 4))
            row("Std dev of ΔW norms",        round(float(np.std(grand_norms)), 4))
            # highest-signal layers
            sorted_stats = sorted(
                [(s[0], s[3]["dw_scaled_norm"]) for s in layer_stats if isinstance(s[3], dict)],
                key=lambda x: -x[1]
            )
            print()
            print("    Top 10 highest-signal layers (likely most trained):")
            for i, (base, norm) in enumerate(sorted_stats[:10]):
                print(f"      {i+1:>2}. {base[-60:]:<60}  ΔW={norm:.4f}")

    else:
        sub("Weight Statistics")
        print("    ⚠  numpy not available — install it for full weight analysis.")
        print("       pip install numpy")

    # ── Parameter count ───────────────────────────────────────────────────
    sub("Parameter Count")
    total_params = 0
    trainable = 0
    for key, info in header.items():
        if not isinstance(info, dict):
            continue
        shape = info.get("shape", [])
        if shape:
            n = 1
            for d in shape: n *= d
            total_params += n
            base, role = parse_lora_key(key)
            if role in ("lora_down", "lora_up", "lora_A", "lora_B"):
                trainable += n

    row("Total params in file",    f"{total_params:,}")
    row("LoRA up/down params",     f"{trainable:,}")
    row("Approx file param dtype", "BF16/F16 (2 bytes each)" if path.stat().st_size < total_params * 3 else "F32 (4 bytes each)")

    # ── Resolution inference attempt ──────────────────────────────────────
    sub("Resolution / Bucket Inference Attempt")
    print("    Scanning conv kernel shapes for spatial dimension hints...")
    spatial_shapes = set()
    for key, info in header.items():
        if not isinstance(info, dict): continue
        shape = info.get("shape", [])
        # Conv lora weights can carry spatial dims
        if len(shape) == 4:
            spatial_shapes.add(tuple(shape))

    if spatial_shapes:
        print("    4D tensor shapes found (possible conv LoRA):")
        for s in sorted(spatial_shapes):
            print(f"      {s}")
    else:
        print("    No 4D (conv) tensors — this is a pure attention/linear LoRA.")
        print("    Resolution and bucket sizes are NOT stored in attention LoRA weights.")
        print()
        print("    ⚠  The bucket sizes used during training exist only in:")
        print("       - The job YAML/TOML you submitted to the cloud service")
        print("       - Training logs (if the service provides them)")
        print()
        print("    What we CAN infer from the weights:")
        print("       - Rank / alpha ✓  (shown above)")
        print("       - Which layers were trained ✓")
        print("       - Relative training signal per layer ✓")
        print("       - Architecture ✓")
        print("       - Effective rank (information capacity) ✓")

    print()


def main():
    cwd = Path(".")
    files = sorted(cwd.glob("*.safetensors"))
    if not files:
        print("No .safetensors files found in current directory.")
        sys.exit(1)
    print(f"Found {len(files)} .safetensors file(s) in {cwd.resolve()}")
    if not HAS_NUMPY:
        print("  ⚠  numpy not found — weight stats disabled. Run: pip install numpy")
    for f in files:
        analyse(f)
    print(f"{'═' * W}\n  Done. {len(files)} file(s).\n{'═' * W}\n")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# NODE API  — called by flux_lora_auto_strength.py
# Returns structured per-layer ΔW data ready for strength computation.
# ─────────────────────────────────────────────────────────────────────────────

def analyse_for_node(path):
    """
    Run the full forensic analysis on a safetensors file and return a dict:
    {
      "db": {
          0: {"img": float, "txt": float},   # mean scaled ΔW per stream
          ...
      },
      "sb": {
          0: float,   # mean scaled ΔW
          ...
      },
      "rank":  int,
      "alpha": float | None,
      "layer_stats": [(base, dw_scaled_norm), ...]   # sorted by base key
    }
    """
    from collections import defaultdict

    path = Path(path)
    meta, header, data_offset = read_header(path)

    # Build layers dict: base_key → {role: tensor_info}
    layers      = defaultdict(dict)
    layer_ranks = {}
    alpha_values = {}

    for key, info in header.items():
        if not isinstance(info, dict):
            continue
        base, role = parse_lora_key(key)
        layers[base][role] = info

    # Read alpha tensor values
    for base, roles in layers.items():
        if "alpha" in roles:
            try:
                raw = read_tensor_bytes(path, roles["alpha"], data_offset)
                arr = bytes_to_floats(raw, roles["alpha"].get("dtype", "F32"))
                if arr is not None and len(arr) > 0:
                    alpha_values[base] = float(arr[0])
            except Exception:
                pass
        dn = roles.get("lora_down") or roles.get("lora_A")
        if dn and "shape" in dn:
            layer_ranks[base] = dn["shape"][0]

    # Compute ΔW scaled norms per layer
    db_img = defaultdict(list)
    db_txt = defaultdict(list)
    sb      = defaultdict(list)
    all_layer_stats = []

    for base in sorted(layers.keys()):
        roles   = layers[base]
        dn_info = roles.get("lora_down") or roles.get("lora_A")
        up_info = roles.get("lora_up")   or roles.get("lora_B")

        if not dn_info or not up_info:
            continue
        if not HAS_NUMPY:
            continue

        try:
            dn_raw = read_tensor_bytes(path, dn_info, data_offset)
            up_raw = read_tensor_bytes(path, up_info, data_offset)
            dn_arr = bytes_to_floats(dn_raw, dn_info.get("dtype", "F32"))
            up_arr = bytes_to_floats(up_raw, up_info.get("dtype", "F32"))
        except Exception:
            continue

        try:
            dn_shape = dn_info["shape"]
            up_shape = up_info["shape"]
            dn_mat   = dn_arr.reshape(dn_shape[0], -1).astype(np.float32)
            up_mat   = up_arr.reshape(up_shape[0], -1).astype(np.float32)
            delta_w  = up_mat @ dn_mat
            dw_norm  = float(np.linalg.norm(delta_w, "fro"))

            alpha    = alpha_values.get(base)
            rank_val = layer_ranks.get(base, 1)
            scale    = (alpha / rank_val) if alpha else 1.0
            dw_scaled = dw_norm * scale
        except Exception:
            continue

        all_layer_stats.append((base, dw_scaled))

        # Classify into db_img / db_txt / sb
        clean = base.replace("diffusion_model.", "")
        parts = clean.split(".")
        try:
            if parts[0] == "double_blocks":
                idx  = int(parts[1])
                rest = ".".join(parts[2:])
                (db_txt if rest.startswith("txt_") else db_img)[idx].append(dw_scaled)
            elif parts[0] == "single_blocks":
                sb[int(parts[1])].append(dw_scaled)
        except (IndexError, ValueError):
            continue

    # Aggregate
    all_alphas = list(alpha_values.values())

    return {
        "db": {
            i: {
                "img": float(np.mean(db_img[i])) if db_img.get(i) else None,
                "txt": float(np.mean(db_txt[i])) if db_txt.get(i) else None,
            }
            for i in range(8)
        },
        "sb": {
            i: float(np.mean(sb[i])) if sb.get(i) else None
            for i in range(24)
        },
        "rank":        sorted(set(layer_ranks.values()))[0] if layer_ranks else 0,
        "alpha":       float(np.mean(all_alphas)) if all_alphas else None,
        "layer_stats": all_layer_stats,
    }


def analyse_for_node_zimage(path):
    """
    Forensic analysis for Z-Image / Lumina2 architecture LoRAs.
    Dynamically discovers which layers are present — no hardcoded range.

    Returns:
    {
      "layers": {
          14: {"attn": float, "ff": float},   # mean scaled ΔW per component
          15: {...},
          ...
      },
      "layer_indices": [14, 15, ...],   # sorted list of discovered layer indices
      "rank":  int,
      "alpha": float | None,
      "layer_stats": [(base, dw_scaled_norm), ...]
    }
    """
    from collections import defaultdict

    path = Path(path)
    meta, header, data_offset = read_header(path)

    layers      = defaultdict(dict)
    layer_ranks = {}
    alpha_values = {}

    for key, info in header.items():
        if not isinstance(info, dict):
            continue
        base, role = parse_lora_key(key)
        layers[base][role] = info

    # Read alpha tensor values from actual bytes
    for base, roles in layers.items():
        if "alpha" in roles:
            try:
                raw = read_tensor_bytes(path, roles["alpha"], data_offset)
                arr = bytes_to_floats(raw, roles["alpha"].get("dtype", "F32"))
                if arr is not None and len(arr) > 0:
                    alpha_values[base] = float(arr[0])
            except Exception:
                pass
        dn = roles.get("lora_down") or roles.get("lora_A")
        if dn and "shape" in dn:
            layer_ranks[base] = dn["shape"][0]

    # Compute ΔW per layer — grouped by layer_idx → attn / ff
    layer_attn = defaultdict(list)
    layer_ff   = defaultdict(list)
    all_layer_stats = []

    for base in sorted(layers.keys()):
        roles   = layers[base]
        dn_info = roles.get("lora_down") or roles.get("lora_A")
        up_info = roles.get("lora_up")   or roles.get("lora_B")

        if not dn_info or not up_info or not HAS_NUMPY:
            continue

        try:
            dn_raw = read_tensor_bytes(path, dn_info, data_offset)
            up_raw = read_tensor_bytes(path, up_info, data_offset)
            dn_arr = bytes_to_floats(dn_raw, dn_info.get("dtype", "F32"))
            up_arr = bytes_to_floats(up_raw, up_info.get("dtype", "F32"))
            dn_mat = dn_arr.reshape(dn_arr.shape[0] if hasattr(dn_arr, 'shape') else layer_ranks.get(base, 1), -1).astype(np.float32)
            up_mat = up_arr.reshape(up_arr.shape[0] if hasattr(up_arr, 'shape') else 1, -1).astype(np.float32)

            # need shape from header
            dn_shape = dn_info["shape"]
            up_shape = up_info["shape"]
            dn_mat = dn_arr.reshape(dn_shape[0], -1).astype(np.float32)
            up_mat = up_arr.reshape(up_shape[0], -1).astype(np.float32)

            delta_w  = up_mat @ dn_mat
            dw_norm  = float(np.linalg.norm(delta_w, "fro"))
            alpha    = alpha_values.get(base)
            rank_val = layer_ranks.get(base, 1)
            scale    = (alpha / rank_val) if alpha else 1.0
            dw_scaled = dw_norm * scale
        except Exception:
            continue

        all_layer_stats.append((base, dw_scaled))

        # Classify — discover layer index and component dynamically
        # Key format: diffusion_model.layers.{N}.attention.* or .feed_forward.*
        clean = base.replace("diffusion_model.", "")
        parts = clean.split(".")
        try:
            if parts[0] == "layers":
                idx = int(parts[1])
                component = parts[2] if len(parts) > 2 else ""
                if component == "attention":
                    layer_attn[idx].append(dw_scaled)
                elif component == "feed_forward":
                    layer_ff[idx].append(dw_scaled)
        except (IndexError, ValueError):
            continue

    # Collect all discovered layer indices
    all_indices = sorted(set(list(layer_attn.keys()) + list(layer_ff.keys())))
    all_alphas  = list(alpha_values.values())

    result_layers = {}
    for idx in all_indices:
        result_layers[idx] = {
            "attn": float(np.mean(layer_attn[idx])) if layer_attn.get(idx) else None,
            "ff":   float(np.mean(layer_ff[idx]))   if layer_ff.get(idx)   else None,
        }

    return {
        "layers":       result_layers,
        "layer_indices": all_indices,
        "rank":         sorted(set(layer_ranks.values()))[0] if layer_ranks else 0,
        "alpha":        float(np.mean(all_alphas)) if all_alphas else None,
        "layer_stats":  all_layer_stats,
    }
