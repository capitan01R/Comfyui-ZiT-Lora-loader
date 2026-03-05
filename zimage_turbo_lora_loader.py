"""
Z-Image Turbo LoRA Loader
Architecture-aware LoRA loading for Z-Image Turbo (Lumina2)

Handles the key mismatches that cause ComfyUI's generic LoRA loader to
silently drop attention weights on Z-Image Turbo models:
  - Fuses separate to_q/to_k/to_v LoRA weights into Z-Image's fused QKV format
  - Remaps to_out.0 -> attention.out
  - Builds architecture-specific key maps via z_image_to_diffusers()
"""

import torch
import comfy.utils
import comfy.lora
import comfy.model_base
import folder_paths
import logging

logger = logging.getLogger(__name__)


def _block_diag(tensors):
    """
    Build a block-diagonal matrix from a list of [out_i, rank_i] tensors.

    For fused QKV lora_B:
      q_up [3840, 64], k_up [3840, 64], v_up [3840, 64]
      -> [11520, 192]

    This is the correct layout so that lora_B @ lora_A reproduces the
    per-head projections without cross-contamination:
      [11520, 192] @ [192, 3840] = [11520, 3840]  matches qkv.weight shape.
    """
    total_out  = sum(t.shape[0] for t in tensors)
    total_rank = sum(t.shape[1] for t in tensors)
    result = torch.zeros(total_out, total_rank, dtype=tensors[0].dtype, device=tensors[0].device)
    row, col = 0, 0
    for t in tensors:
        result[row:row + t.shape[0], col:col + t.shape[1]] = t
        row += t.shape[0]
        col += t.shape[1]
    return result


class ZImageTurboLoraLoader:
    """
    Specialized LoRA loader for Z-Image Turbo (Lumina2 architecture).

    Z-Image Turbo Architecture:
    - 30 transformer layers with fused QKV attention
    - dim=3840, n_heads=30, n_kv_heads=30
    - attention.qkv [11520, 3840] instead of separate to_q/to_k/to_v
    - attention.out instead of to_out.0
    - SwiGLU feed-forward with w1/w2/w3
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.01,
                }),
                "auto_convert_qkv": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto-convert Q/K/V -> fused QKV",
                    "label_off": "Direct load (no conversion)",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/Z-Image"
    TITLE = "Z-Image Turbo LoRA Loader"

    def load_lora(self, model, lora_name, strength_model, auto_convert_qkv=True):
        if strength_model == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        logger.info(f"[Z-Image LoRA] Loading: {lora_name} ({len(lora_sd)} keys)")

        if not isinstance(model.model, comfy.model_base.Lumina2):
            logger.warning(
                f"[Z-Image LoRA] Model is {type(model.model).__name__}, not Lumina2/Z-Image Turbo. "
                "Key mapping may not work correctly."
            )

        # Convert separate Q/K/V -> fused QKV if the LoRA needs it
        if auto_convert_qkv and self._has_separate_qkv(lora_sd):
            logger.info("[Z-Image LoRA] Detected separate Q/K/V -- fusing to QKV format")
            lora_sd = self._convert_to_fused_qkv(lora_sd, model)

        # Build key map AFTER conversion so fused keys are included
        key_map = self._build_key_map(model, lora_sd)

        # Load and apply patches
        patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)

        loaded = set()
        for x in key_map:
            for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha",
                           ".lora_down.weight", ".lora_up.weight",
                           ".diff", ".diff_b", ".w_norm", ".b_norm"):
                if f"{x}{suffix}" in lora_sd:
                    loaded.add(x)
                    break

        logger.info(f"[Z-Image LoRA] Key map: {len(key_map)}, LoRA keys found: {len(loaded)}, "
                    f"patches created: {len(patch_dict)} (strength={strength_model})")

        model_lora = model.clone()
        model_lora.add_patches(patch_dict, strength_patch=strength_model, strength_model=1.0)

        return (model_lora,)

    # ------------------------------------------------------------------
    # Key mapping
    # ------------------------------------------------------------------

    def _build_key_map(self, model, lora_sd=None):
        """
        Build Z-Image Turbo key mapping.

        Combines z_image_to_diffusers() mappings with direct native key
        identity mappings so that BOTH diffusers-format AND native-format
        LoRA keys can be resolved.
        """
        key_map = {}

        # Step 1: native identity mappings
        for model_key in model.model.state_dict().keys():
            if model_key.startswith("diffusion_model.") and model_key.endswith(".weight"):
                base = model_key[:-len(".weight")]
                key_map[base] = model_key
                if base.startswith("diffusion_model."):
                    key_map[base[len("diffusion_model."):]] = model_key

        # Step 2: z_image_to_diffusers() mappings (diffusers -> native)
        if isinstance(model.model, comfy.model_base.Lumina2):
            try:
                diffusers_keys = comfy.utils.z_image_to_diffusers(
                    model.model.model_config.unet_config,
                    output_prefix="diffusion_model.",
                )
                for k, target in diffusers_keys.items():
                    if not k.endswith(".weight"):
                        continue
                    lora_key = k[:-len(".weight")]
                    key_map[f"diffusion_model.{lora_key}"] = target
                    key_map[f"transformer.{lora_key}"] = target
                    key_map[f"lycoris_{lora_key.replace('.', '_')}"] = target
                    key_map[lora_key] = target
                logger.info(f"[Z-Image LoRA] z_image_to_diffusers added entries, total: {len(key_map)}")
            except Exception as e:
                logger.warning(f"[Z-Image LoRA] z_image_to_diffusers failed ({e})")

        # Step 3: catch any remaining LoRA keys via direct candidate lookup
        if lora_sd is not None:
            model_sd_keys = set(model.model.state_dict().keys())
            for k in lora_sd:
                for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha",
                               ".lora_down.weight", ".lora_up.weight",
                               ".dora_scale", ".diff", ".diff_b"):
                    if k.endswith(suffix):
                        prefix = k[:-len(suffix)]
                        if prefix not in key_map:
                            for c in (f"{prefix}.weight", f"diffusion_model.{prefix}.weight"):
                                if c in model_sd_keys:
                                    key_map[prefix] = c
                                    break
                        break

        return key_map

    # ------------------------------------------------------------------
    # QKV fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _has_separate_qkv(lora_sd):
        return any(
            ".to_q.lora_A" in k or ".to_k.lora_A" in k or ".to_v.lora_A" in k
            for k in lora_sd
        )

    @staticmethod
    def _convert_to_fused_qkv(lora_sd, model):
        """
        Fuse separate to_q / to_k / to_v LoRA weights into a single qkv tensor
        and rename to_out.0 -> out to match Z-Image Turbo's architecture.

        Correct fusion layout:
          lora_A: cat([q_down, k_down, v_down], dim=0)  -> [rank*3, dim]
          lora_B: block_diag(q_up, k_up, v_up)          -> [dim*3,  rank*3]

        lora_B must be block-diagonal (NOT concatenated along dim=0) so that
          lora_B @ lora_A = [dim*3, rank*3] @ [rank*3, dim] = [dim*3, dim]
        which matches qkv.weight [11520, 3840].

        Concatenating lora_B along dim=0 produces [dim*3, rank] which cannot
        be multiplied with lora_A [rank*3, dim] -> shape error at inference.
        """
        converted = {}
        processed = set()

        # Collect all base paths that have Q/K/V
        qkv_groups = {}
        for k in lora_sd:
            for component in ("to_q", "to_k", "to_v"):
                tag = f".{component}.lora_A.weight"
                if k.endswith(tag):
                    base = k[:-len(tag)]
                    if base not in qkv_groups:
                        qkv_groups[base] = {}
                    down_key = f"{base}.{component}.lora_A.weight"
                    up_key   = f"{base}.{component}.lora_B.weight"
                    if down_key in lora_sd and up_key in lora_sd:
                        qkv_groups[base][component] = (lora_sd[down_key], lora_sd[up_key])
                    break

        # Fuse Q/K/V groups
        for base, parts in qkv_groups.items():
            if not (len(parts) == 3 and all(c in parts for c in ("to_q", "to_k", "to_v"))):
                continue

            q_down, q_up = parts["to_q"]
            k_down, k_up = parts["to_k"]
            v_down, v_up = parts["to_v"]

            # lora_A: stack down projections -> [rank*3, dim]
            fused_A = torch.cat([q_down, k_down, v_down], dim=0)

            # lora_B: block-diagonal of up projections -> [dim*3, rank*3]
            fused_B = _block_diag([q_up, k_up, v_up])

            converted[f"{base}.qkv.lora_A.weight"] = fused_A
            converted[f"{base}.qkv.lora_B.weight"] = fused_B

            logger.debug(f"[Z-Image LoRA] Fused QKV: {base} "
                         f"A={list(fused_A.shape)} B={list(fused_B.shape)}")

            # Alpha: scale to match the new effective rank (rank*3)
            alphas = []
            for component in ("to_q", "to_k", "to_v"):
                alpha_key = f"{base}.{component}.alpha"
                if alpha_key in lora_sd:
                    alphas.append(lora_sd[alpha_key])
                    processed.add(alpha_key)
            if alphas:
                # original alpha/rank preserved: new_alpha = avg_alpha * 3 / 3 = avg_alpha
                # but effective rank is now rank*3, so keep alpha = avg to preserve alpha/rank ratio
                converted[f"{base}.qkv.alpha"] = sum(alphas) / len(alphas)

            for component in ("to_q", "to_k", "to_v"):
                processed.add(f"{base}.{component}.lora_A.weight")
                processed.add(f"{base}.{component}.lora_B.weight")

        # Rename to_out.0 -> out
        out_bases = set()
        for k in lora_sd:
            if ".to_out.0.lora_A.weight" in k:
                out_bases.add(k[:k.index(".to_out.0.lora_A.weight")])

        for base in out_bases:
            for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha"):
                old = f"{base}.to_out.0{suffix}"
                new = f"{base}.out{suffix}"
                if old in lora_sd and old not in processed:
                    converted[new] = lora_sd[old]
                    processed.add(old)
                    logger.debug(f"[Z-Image LoRA] Remapped: {old} -> {new}")

        # Pass through everything else untouched
        for key, value in lora_sd.items():
            if key not in processed:
                converted[key] = value

        fused_keys = [k for k in converted if ".qkv.lora_" in k]
        logger.info(f"[Z-Image LoRA] Converted {len(processed)} keys -> {len(converted)} total, "
                    f"{len(fused_keys)} fused QKV tensors")
        for k in sorted(fused_keys)[:6]:
            logger.info(f"[Z-Image LoRA]   {k}  {list(converted[k].shape)}")

        return converted


NODE_CLASS_MAPPINGS = {
    "ZImageTurboLoraLoader": ZImageTurboLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboLoraLoader": "Z-Image Turbo LoRA Loader",
}
