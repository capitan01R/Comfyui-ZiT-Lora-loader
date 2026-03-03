# ComfyUI Z-Image Turbo LoRA Loader
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


Architecture-aware LoRA loader for **Z-Image Turbo** in ComfyUI.
![](images/node.png)

## Why not the default LoRA loader?

Z-Image Turbo uses a non-standard attention architecture that breaks ComfyUI's generic LoRA loader:

| Feature | Generic Loader | This Node |
|---|---|---|
| **Fused QKV attention** | Can't map separate `to_q`/`to_k`/`to_v` to fused `attention.qkv` -- silently drops keys | Auto-fuses Q/K/V into the correct `[11520, 3840]` tensor |
| **Output projection** | Expects `to_out.0`, Z-Image uses `attention.out` -- keys dropped | Remaps automatically |
| **Key mapping** | Generic heuristics | Uses `z_image_to_diffusers()` for exact Lumina2 architecture mapping |
| **Multi-format support** | Limited | Handles `diffusion_model.*`, `transformer.*`, `lycoris_*`, and bare prefixes |

The result: with the generic loader, most or all attention LoRA weights are silently dropped. This node ensures they actually get applied.

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/capitan01R/Comfyui-ZiT-Lora-loader.git
```

No additional dependencies required -- uses only ComfyUI builtins.

## Usage

The node appears under **loaders/Z-Image** as **Z-Image Turbo LoRA Loader**

### Inputs

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | Z-Image Turbo / Lumina2 model |
| `lora_name` | dropdown | LoRA file from your `models/loras` folder |
| `strength_model` | float | LoRA strength (-20.0 to 20.0, default 1.0) |
| `auto_convert_qkv` | boolean | Automatically fuse separate Q/K/V to Z-Image's fused QKV format (default: on) |

### When to enable auto_convert_qkv

- **ON** (default): If your LoRA was trained with a diffusers-style trainer that produces separate `to_q`/`to_k`/`to_v` weights
- **OFF**: If your LoRA was already trained against Z-Image Turbo's native fused QKV format

## Z-Image Turbo Architecture Reference

```
30 transformer layers
  attention
    qkv.weight      [11520, 3840]  (fused Q+K+V)
    out.weight       [3840, 3840]
    q_norm.weight    [128]
    k_norm.weight    [128]
  feed_forward
    w1.weight        [10240, 3840]  (SwiGLU)
    w2.weight        [3840, 10240]
    w3.weight        [10240, 3840]
  attention_norm.weight [3840]
  ffn_norm.weight       [3840]
  modulation.linear.weight [15360, 3840]  (AdaLN)
```

dim=3840, n_heads=30, n_kv_heads=30, head_dim=128

MIT
