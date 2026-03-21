"""
zimage_lora_auto_strength.py

Same forensic auto-strength treatment as flux_lora_auto_strength.py,
but for Z-Image Turbo (Lumina2) architecture.

Uses lora_meta.analyse_for_node_zimage() — discovers layers dynamically,
no hardcoded layer range.

ZImageLoraAutoStrength  — outputs layer_strengths JSON + lora_name.
                          Wire to ZImageTurboLoraLoader. One knob: global_strength.

ZImageLoraAutoLoader    — fully self-contained. model in → patched model out.
                          One knob: global_strength.
"""

import json
import logging
import numpy as np
import folder_paths
import comfy.utils
import comfy.lora

from .lora_meta import analyse_for_node_zimage

logger = logging.getLogger(__name__)

_FLOOR   = 0.30
_CEILING = 1.50


# ─────────────────────────────────────────────────────────────────────────────
# STRENGTH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _all_norms(analysis):
    out = []
    for idx in analysis["layer_indices"]:
        d = analysis["layers"].get(idx, {})
        if d.get("attn") is not None: out.append(d["attn"])
        if d.get("ff")   is not None: out.append(d["ff"])
    return out


def compute_strengths(analysis, global_strength):
    all_norms = _all_norms(analysis)
    if not all_norms:
        return {str(i): {"attn": global_strength, "ff": global_strength}
                for i in analysis["layer_indices"]}

    mean_norm = float(np.mean(all_norms))

    def clamp(v):
        return max(_FLOOR, min(_CEILING, v))

    def map_norm(norm):
        if norm is None or norm < 1e-8:
            return global_strength
        return clamp(global_strength * (mean_norm / norm))

    out = {}
    for idx in analysis["layer_indices"]:
        d = analysis["layers"].get(idx, {})
        out[str(idx)] = {
            "attn": round(map_norm(d.get("attn")), 4),
            "ff":   round(map_norm(d.get("ff")),   4),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_report(lora_name, analysis, strengths, global_strength):
    all_norms = _all_norms(analysis)
    median    = float(np.median(all_norms)) if all_norms else 1.0
    alpha_str = f"{analysis['alpha']:.4f}" if analysis['alpha'] is not None else "1.0 (not embedded)"

    lines = [
        "═══ Z-Image LoRA Auto Strength ═════════════════════",
        f"  LoRA    : {lora_name}",
        f"  Rank    : {analysis['rank']}   Alpha: {alpha_str}",
        f"  Global  : {global_strength}",
        f"  Layers  : {analysis['layer_indices']}",
    ]
    if all_norms:
        lines.append(
            f"  ΔW      : mean={np.mean(all_norms):.3f}  "
            f"median={median:.3f}  max={np.max(all_norms):.3f}"
        )
    lines += ["", "  LAYERS", "  " + "─" * 52]
    for idx in analysis["layer_indices"]:
        d  = analysis["layers"].get(idx, {})
        s  = strengths.get(str(idx), {})
        an = d.get("attn") or 0.0
        fn = d.get("ff")   or 0.0
        hot_a = " ★" if (d.get("attn") and d["attn"] > median * 1.5) else ""
        hot_f = " ★" if (d.get("ff")   and d["ff"]   > median * 1.5) else ""
        lines.append(
            f"  [{idx:>2}] attn ΔW={an:.3f}→{s.get('attn', 0):.4f}{hot_a}   "
            f"ff ΔW={fn:.3f}→{s.get('ff', 0):.4f}{hot_f}"
        )
    lines.append("═════════════════════════════════════════════════════")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED APPLY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def apply_layer_strengths(lora_sd, strengths_cfg, global_strength):
    """
    Scale lora_B tensors by per-layer strength.
    strengths_cfg: {str(layer_idx): {"attn": float, "ff": float}}
    """
    if not strengths_cfg or abs(global_strength) < 1e-8:
        return lora_sd

    scaled = {}
    for key, tensor in lora_sd.items():
        if not (key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight")):
            scaled[key] = tensor
            continue

        parts     = key.split(".")
        layer_idx = None
        component = None

        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
            if p == "attention":
                component = "attn"
            elif p == "feed_forward":
                component = "ff"

        if layer_idx is not None and component is not None:
            cfg    = strengths_cfg.get(str(layer_idx), {})
            target = cfg.get(component)
            if target is not None:
                scale = target / global_strength if abs(global_strength) > 1e-8 else target
                scaled[key] = tensor * scale
                continue

        scaled[key] = tensor

    return scaled


def build_key_map(model):
    key_map = {}
    for model_key in model.model.state_dict().keys():
        if not model_key.endswith(".weight"):
            continue
        base = model_key[: -len(".weight")]
        bare = base[len("diffusion_model."):] if base.startswith("diffusion_model.") else base
        for pfx in ("diffusion_model.", "transformer.", ""):
            key_map[f"{pfx}{bare}"] = model_key
        key_map["lora_unet_" + bare.replace(".", "_")] = model_key
    return key_map


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — ZImageLoraAutoStrength
# ─────────────────────────────────────────────────────────────────────────────

class ZImageLoraAutoStrength:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name":       (folder_paths.get_filename_list("loras"),),
            "global_strength": ("FLOAT", {
                "default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01,
                "tooltip": "Master strength. All per-layer values are auto-computed from ΔW forensics.",
            }),
        }}

    RETURN_TYPES  = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES  = ("layer_strengths", "analysis_report", "global_strength", "lora_name")
    FUNCTION      = "run"
    CATEGORY      = "loaders/Z-Image"
    TITLE         = "Z-Image LoRA Auto Strength"

    def run(self, lora_name, global_strength):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        analysis  = analyse_for_node_zimage(lora_path)
        strengths = compute_strengths(analysis, global_strength)
        report    = build_report(lora_name, analysis, strengths, global_strength)
        logger.info(f"[ZImageAutoStrength] {lora_name} — layers={analysis['layer_indices']} rank={analysis['rank']}")
        return (json.dumps(strengths), report, global_strength, lora_name)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — ZImageLoraAutoLoader
# ─────────────────────────────────────────────────────────────────────────────

class ZImageLoraAutoLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":           ("MODEL",),
            "lora_name":       (folder_paths.get_filename_list("loras"),),
            "global_strength": ("FLOAT", {
                "default": 0.75, "min": -2.0, "max": 2.0, "step": 0.01,
                "tooltip": "Master strength. Everything else is computed automatically.",
            }),
        }}

    RETURN_TYPES  = ("MODEL", "STRING")
    RETURN_NAMES  = ("model", "analysis_report")
    FUNCTION      = "run"
    CATEGORY      = "loaders/Z-Image"
    TITLE         = "Z-Image LoRA Auto Loader"

    def run(self, model, lora_name, global_strength):
        if global_strength == 0:
            return (model, "Skipped — global_strength is 0.")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        analysis  = analyse_for_node_zimage(lora_path)
        strengths = compute_strengths(analysis, global_strength)
        report    = build_report(lora_name, analysis, strengths, global_strength)

        sd         = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # Verify scaling is applied
        sample_key = next((k for k in sd if k.endswith(".lora_B.weight") or k.endswith(".lora_up.weight")), None)
        norm_before = float(sd[sample_key].float().norm().item()) if sample_key else None

        sd_scaled  = apply_layer_strengths(sd, strengths, global_strength)

        norm_after = float(sd_scaled[sample_key].float().norm().item()) if sample_key else None
        if norm_before is not None:
            ratio = norm_after / norm_before if norm_before > 1e-8 else 0
            logger.warning(
                f"[ZImageAutoLoader] ✅ AUTO-STRENGTH APPLIED — "
                f"'{sample_key}' norm: {norm_before:.6f} → {norm_after:.6f} (ratio={ratio:.4f})"
            )

        key_map    = build_key_map(model)
        patch_dict = comfy.lora.load_lora(sd_scaled, key_map, log_missing=False)
        model_out  = model.clone()
        model_out.add_patches(patch_dict, strength_patch=1.0, strength_model=1.0)

        logger.info(f"[ZImageAutoLoader] {lora_name} — {len(patch_dict)} patches @ global={global_strength}")
        return (model_out, report)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ZImageLoraAutoStrength": ZImageLoraAutoStrength,
    "ZImageLoraAutoLoader":   ZImageLoraAutoLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoraAutoStrength": "Z-Image LoRA Auto Strength",
    "ZImageLoraAutoLoader":   "Z-Image LoRA Auto Loader",
}
