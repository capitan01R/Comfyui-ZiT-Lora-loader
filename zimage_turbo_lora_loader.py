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

        # Build architecture-specific key map
        key_map = self._build_key_map(model)

        # Convert separate Q/K/V -> fused QKV if the LoRA needs it
        if auto_convert_qkv and self._has_separate_qkv(lora_sd):
            logger.info("[Z-Image LoRA] Detected separate Q/K/V -- fusing to QKV format")
            lora_sd = self._convert_to_fused_qkv(lora_sd, model)

        # Load and apply patches
        patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
        logger.info(f"[Z-Image LoRA] Applied {len(patch_dict)} patches (strength={strength_model})")

        model_lora = model.clone()
        model_lora.add_patches(patch_dict, strength_patch=strength_model, strength_model=1.0)

        return (model_lora,)

    # ------------------------------------------------------------------
    # Key mapping
    # ------------------------------------------------------------------

    def _build_key_map(self, model):
        """
        Build Z-Image Turbo key mapping.

        Tries ComfyUI's built-in z_image_to_diffusers() first (handles the full
        Lumina2 architecture), falls back to manual state_dict iteration.
        """
        key_map = {}

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

                    # Accept LoRA keys from all common naming conventions
                    key_map[f"diffusion_model.{lora_key}"] = target
                    key_map[f"transformer.{lora_key}"] = target
                    key_map[f"lycoris_{lora_key.replace('.', '_')}"] = target
                    key_map[lora_key] = target

                logger.info(f"[Z-Image LoRA] Key map: {len(key_map)} entries (z_image_to_diffusers)")
                return key_map

            except Exception as e:
                logger.warning(f"[Z-Image LoRA] z_image_to_diffusers failed ({e}), using fallback")

        # Fallback: derive from model state dict
        for model_key in model.model.state_dict().keys():
            if model_key.startswith("diffusion_model.") and model_key.endswith(".weight"):
                base = model_key[len("diffusion_model."):-len(".weight")]
                key_map[base] = model_key
                key_map[f"diffusion_model.{base}"] = model_key
                key_map[f"transformer.{base}"] = model_key

        logger.info(f"[Z-Image LoRA] Key map: {len(key_map)} entries (fallback)")
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
        n_layers = getattr(model.model.model_config.unet_config, "n_layers", 30)

        converted = {}
        processed = set()

        for layer_idx in range(n_layers):
            for prefix in ("diffusion_model.", "transformer.", ""):
                base = f"{prefix}layers.{layer_idx}.attention"

                # --- Fuse Q/K/V -------------------------------------------
                qkv_parts = {}
                for component in ("to_q", "to_k", "to_v"):
                    down_key = f"{base}.{component}.lora_A.weight"
                    up_key = f"{base}.{component}.lora_B.weight"
                    if down_key in lora_sd and up_key in lora_sd:
                        qkv_parts[component] = (lora_sd[down_key], lora_sd[up_key])

                if len(qkv_parts) == 3:
                    q_down, q_up = qkv_parts["to_q"]
                    k_down, k_up = qkv_parts["to_k"]
                    v_down, v_up = qkv_parts["to_v"]

                    converted[f"{base}.qkv.lora_A.weight"] = torch.cat([q_down, k_down, v_down], dim=0)
                    converted[f"{base}.qkv.lora_B.weight"] = torch.cat([q_up, k_up, v_up], dim=0)

                    # Average alpha values
                    alphas = []
                    for component in ("to_q", "to_k", "to_v"):
                        alpha_key = f"{base}.{component}.alpha"
                        if alpha_key in lora_sd:
                            alphas.append(lora_sd[alpha_key])
                            processed.add(alpha_key)
                    if len(alphas) == 3:
                        converted[f"{base}.qkv.alpha"] = sum(alphas) / 3.0

                    for component in ("to_q", "to_k", "to_v"):
                        processed.add(f"{base}.{component}.lora_A.weight")
                        processed.add(f"{base}.{component}.lora_B.weight")

                    logger.debug(f"[Z-Image LoRA] Layer {layer_idx}: fused QKV")
                    break  # done with this layer

                # --- Rename to_out.0 -> out --------------------------------
                out_down_key = f"{base}.to_out.0.lora_A.weight"
                out_up_key = f"{base}.to_out.0.lora_B.weight"

                if out_down_key in lora_sd and out_up_key in lora_sd:
                    converted[f"{base}.out.lora_A.weight"] = lora_sd[out_down_key]
                    converted[f"{base}.out.lora_B.weight"] = lora_sd[out_up_key]
                    processed.update([out_down_key, out_up_key])

                    out_alpha_key = f"{base}.to_out.0.alpha"
                    if out_alpha_key in lora_sd:
                        converted[f"{base}.out.alpha"] = lora_sd[out_alpha_key]
                        processed.add(out_alpha_key)

                    logger.debug(f"[Z-Image LoRA] Layer {layer_idx}: remapped out projection")
                    break

        # Pass through everything else untouched
        for key, value in lora_sd.items():
            if key not in processed:
                converted[key] = value

        logger.info(f"[Z-Image LoRA] Converted {len(processed)} keys to Z-Image format")
        return converted


_MAX_SLOTS = 10


class ZImageTurboLoraStack(ZImageTurboLoraLoader):

    @classmethod
    def INPUT_TYPES(cls):
        lora_opts = ["None"] + folder_paths.get_filename_list("loras")
        optional = {}
        for i in range(1, _MAX_SLOTS + 1):
            optional[f"lora_{i}"]     = (lora_opts,)
            optional[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01})
            optional[f"enabled_{i}"]  = ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"})
            optional[f"fuse_qkv_{i}"] = ("BOOLEAN", {"default": True, "label_on": "Fuse QKV", "label_off": "Direct"})
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "loaders/Z-Image"
    TITLE = "Z-Image Turbo LoRA Stack"

    def load_loras(self, model, **kwargs):
        if not isinstance(model.model, comfy.model_base.Lumina2):
            logger.warning(
                f"[Z-Image LoRA Stack] Model is {type(model.model).__name__}, not Lumina2/Z-Image Turbo. "
                "Key mapping may not work correctly."
            )

        key_map = self._build_key_map(model)
        current = model

        for i in range(1, _MAX_SLOTS + 1):
            name     = kwargs.get(f"lora_{i}", "None")
            strength = kwargs.get(f"strength_{i}", 1.0)
            enabled  = kwargs.get(f"enabled_{i}", True)
            fuse_qkv = kwargs.get(f"fuse_qkv_{i}", True)

            if not enabled or name == "None" or strength == 0:
                continue

            lora_path = folder_paths.get_full_path("loras", name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            logger.info(f"[Z-Image LoRA Stack] Slot {i}: {name} ({len(lora_sd)} keys, strength={strength}, fuse_qkv={fuse_qkv})")

            if fuse_qkv and self._has_separate_qkv(lora_sd):
                lora_sd = self._convert_to_fused_qkv(lora_sd, current)

            patch_dict = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
            logger.info(f"[Z-Image LoRA Stack] Slot {i}: applied {len(patch_dict)} patches")

            next_model = current.clone()
            next_model.add_patches(patch_dict, strength_patch=strength, strength_model=1.0)
            current = next_model

        return (current,)


NODE_CLASS_MAPPINGS = {
    "ZImageTurboLoraLoader": ZImageTurboLoraLoader,
    "ZImageTurboLoraStack":  ZImageTurboLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboLoraLoader": "Z-Image Turbo LoRA Loader",
    "ZImageTurboLoraStack":  "Z-Image Turbo LoRA Stack",
}
