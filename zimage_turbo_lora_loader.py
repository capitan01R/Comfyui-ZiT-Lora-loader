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
        logger.info(f"[Z-Image LoRA] Applied {len(patch_dict)} patches (strength={strength_model})")

        # Log what we missed
        loaded = set()
        for x in key_map:
            for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha",
                           ".lora_down.weight", ".lora_up.weight",
                           ".diff", ".diff_b", ".w_norm", ".b_norm"):
                if f"{x}{suffix}" in lora_sd:
                    loaded.add(x)
                    break

        matched = set()
        for x in key_map:
            if key_map[x] in patch_dict or (isinstance(key_map[x], tuple) and key_map[x][0] in patch_dict):
                matched.add(x)

        logger.info(f"[Z-Image LoRA] Key map entries: {len(key_map)}, "
                     f"LoRA keys found: {len(loaded)}, patches created: {len(patch_dict)}")

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

        # Step 1: Always add native identity mappings from the model state dict.
        # This ensures that LoRA keys already in native format (like after
        # our QKV fusion) can find their target weights.
        for model_key in model.model.state_dict().keys():
            if model_key.startswith("diffusion_model.") and model_key.endswith(".weight"):
                base = model_key[:-len(".weight")]
                # Native key -> native weight
                key_map[base] = model_key
                # Also accept without prefix
                if base.startswith("diffusion_model."):
                    short = base[len("diffusion_model."):]
                    key_map[short] = model_key

        # Step 2: Add z_image_to_diffusers() mappings on top (diffusers -> native).
        # These handle LoRAs that use diffusers-style key naming.
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

                logger.info(f"[Z-Image LoRA] Key map: {len(key_map)} entries")

            except Exception as e:
                logger.warning(f"[Z-Image LoRA] z_image_to_diffusers failed ({e})")

        # Step 3: If we have the actual LoRA state dict, add any prefixed
        # variants we see in it to maximize matching.
        if lora_sd is not None:
            lora_prefixes = set()
            for k in lora_sd:
                # Strip suffixes like .lora_A.weight, .alpha, etc
                for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha",
                               ".lora_down.weight", ".lora_up.weight",
                               ".dora_scale", ".diff", ".diff_b"):
                    if k.endswith(suffix):
                        lora_prefixes.add(k[:-len(suffix)])
                        break

            model_sd_keys = set(model.model.state_dict().keys())
            for prefix in lora_prefixes:
                if prefix in key_map:
                    continue
                # Try to find the matching model weight
                candidates = [
                    f"{prefix}.weight",
                    f"diffusion_model.{prefix}.weight",
                ]
                for c in candidates:
                    if c in model_sd_keys:
                        key_map[prefix] = c
                        break

        return key_map

    # ------------------------------------------------------------------
    # QKV fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _has_separate_qkv(lora_sd):
        """Check if LoRA uses separate to_q/to_k/to_v instead of fused qkv."""
        return any(
            ".to_q.lora_A" in k or ".to_k.lora_A" in k or ".to_v.lora_A" in k
            for k in lora_sd
        )

    @staticmethod
    def _convert_to_fused_qkv(lora_sd, model):
        """
        Fuse separate to_q / to_k / to_v LoRA weights into a single qkv tensor
        and rename to_out.0 -> out to match Z-Image Turbo's architecture.

        Concatenation layout (matches Z-Image's fused QKV):
          lora_A: cat([q_down, k_down, v_down], dim=0)  ->  [rank*3, dim]
          lora_B: cat([q_up,   k_up,   v_up],   dim=0)  ->  [dim*3,  rank]
        """
        converted = {}
        processed = set()

        # Collect all layer indices and prefixes that have Q/K/V
        qkv_groups = {}  # (prefix, layer_idx) -> {component: (down, up)}
        out_renames = {}  # old_base -> new_base

        for k in lora_sd:
            for component in ("to_q", "to_k", "to_v"):
                if f".{component}.lora_A" in k:
                    # Extract the base path before .to_q etc
                    idx = k.index(f".{component}.")
                    base = k[:idx]
                    if base not in qkv_groups:
                        qkv_groups[base] = {}

                    down_key = f"{base}.{component}.lora_A.weight"
                    up_key = f"{base}.{component}.lora_B.weight"

                    if down_key in lora_sd and up_key in lora_sd:
                        qkv_groups[base][component] = (lora_sd[down_key], lora_sd[up_key])
                    break

            # Track to_out.0 renames
            if ".to_out.0.lora_A" in k:
                idx = k.index(".to_out.0.")
                base = k[:idx]
                out_renames[base] = True

        # Fuse Q/K/V groups
        for base, parts in qkv_groups.items():
            if len(parts) == 3 and "to_q" in parts and "to_k" in parts and "to_v" in parts:
                q_down, q_up = parts["to_q"]
                k_down, k_up = parts["to_k"]
                v_down, v_up = parts["to_v"]

                # Figure out the correct native key path
                # LoRA might use: diffusion_model.layers.X.attention.to_q
                # Native model uses: diffusion_model.layers.X.attention.qkv
                native_base = base
                # Strip any attention suffix variations
                if native_base.endswith(".attention"):
                    pass  # already correct
                elif ".attention" in native_base:
                    pass  # has attention in path

                converted[f"{native_base}.qkv.lora_A.weight"] = torch.cat([q_down, k_down, v_down], dim=0)
                converted[f"{native_base}.qkv.lora_B.weight"] = torch.cat([q_up, k_up, v_up], dim=0)

                # Average alpha values
                alphas = []
                for component in ("to_q", "to_k", "to_v"):
                    alpha_key = f"{base}.{component}.alpha"
                    if alpha_key in lora_sd:
                        alphas.append(lora_sd[alpha_key])
                        processed.add(alpha_key)
                if alphas:
                    converted[f"{native_base}.qkv.alpha"] = sum(alphas) / len(alphas)

                for component in ("to_q", "to_k", "to_v"):
                    processed.add(f"{base}.{component}.lora_A.weight")
                    processed.add(f"{base}.{component}.lora_B.weight")

                logger.debug(f"[Z-Image LoRA] Fused QKV: {base} -> {native_base}.qkv")

        # Rename to_out.0 -> out
        for base in out_renames:
            for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha"):
                old_key = f"{base}.to_out.0{suffix}"
                new_key = f"{base}.out{suffix}"
                if old_key in lora_sd and old_key not in processed:
                    converted[new_key] = lora_sd[old_key]
                    processed.add(old_key)
                    logger.debug(f"[Z-Image LoRA] Remapped: {old_key} -> {new_key}")

        # Pass through everything else untouched
        for key, value in lora_sd.items():
            if key not in processed:
                converted[key] = value

        logger.info(f"[Z-Image LoRA] Converted {len(processed)} keys, "
                     f"output has {len(converted)} keys")

        # Debug: log the fused keys we created
        fused_keys = [k for k in converted if ".qkv.lora_" in k]
        logger.info(f"[Z-Image LoRA] Fused QKV keys created: {len(fused_keys)}")
        for k in sorted(fused_keys)[:5]:
            logger.info(f"[Z-Image LoRA]   {k}  {list(converted[k].shape)}")

        return converted


NODE_CLASS_MAPPINGS = {
    "ZImageTurboLoraLoader": ZImageTurboLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboLoraLoader": "Z-Image Turbo LoRA Loader",
}
