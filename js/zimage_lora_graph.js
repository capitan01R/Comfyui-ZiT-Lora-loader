// Z-Image Turbo LoRA Loader — Layer Strength Graph Widget
//
// Shows 30 columns (one per transformer layer, layers 0-29).
// Each column has two draggable bars:
//   TOP (purple)  = attention strength  (Q/K/V/out)
//   BOTTOM (teal) = feed-forward strength (w2/w3)
//
// Bar height maps 0.0 → 2.0 (global strength shown as a reference line).
// Layers with no LoRA weights are dimmed automatically after the first run.
//
// Controls:
//   Drag bar up/down   → set that component's strength
//   Click bar          → toggle between 0 and last non-zero value
//   Shift-drag         → move all active layers together
//   Buttons: Reset All | Mirror (copy attn→ff) | Flatten (set all to global)
//
// Serializes to hidden `layer_strengths` widget as JSON:
//   { "14": { "attn": 1.2, "ff": 0.8 }, "15": { ... }, ... }

import { app } from "../../scripts/app.js";

const N_LAYERS  = 30;
const STR_MAX   = 2.0;
const STR_MIN   = 0.0;

const PAD       = 10;
const GRAPH_H   = 160;   // bar chart height
const BTN_ROW_H = 26;
const LABEL_H   = 18;
const WIDGET_H  = GRAPH_H + BTN_ROW_H + LABEL_H + PAD * 3;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function hideWidget(node, widget) {
    if (!widget) return;
    widget.type        = "hidden_zimage";
    widget.computeSize = () => [0, -4];
}

// Default: all layers at 1.0 for both attn and ff
function defaultStrengths() {
    const s = {};
    for (let i = 0; i < N_LAYERS; i++) s[i] = { attn: 1.0, ff: 1.0 };
    return s;
}

app.registerExtension({
    name: "Comfy.ZImageLoraGraph",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ZImageTurboLoraLoader") return;

        const _onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = _onNodeCreated?.apply(this, arguments);
            const node   = this;
            const W      = (name) => node.widgets?.find(w => w.name === name);

            setTimeout(() => {
                hideWidget(node, W("layer_strengths"));
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }, 0);

            // ── State ─────────────────────────────────────────────────────────
            let strengths   = defaultStrengths();
            // Which layers actually have LoRA weights (detected from saved JSON or assumed all)
            let activeLayers = new Set(Array.from({ length: N_LAYERS }, (_, i) => i));

            // Track last non-zero value for toggle
            let lastAttn = Array(N_LAYERS).fill(1.0);
            let lastFf   = Array(N_LAYERS).fill(1.0);

            // ── Auto-strength live watcher ─────────────────────────────────────
            // Parses incoming layer_strengths JSON (from ZImageLoraAutoStrength)
            // and pushes values into the bar state immediately.
            function applyStrengthJSON(value) {
                try {
                    if (!value || value === "{}") return;
                    const raw = JSON.parse(value);
                    if (typeof raw !== "object") return;
                    // Reset active layers to only those present in the JSON
                    activeLayers = new Set();
                    for (const [k, v] of Object.entries(raw)) {
                        const idx = parseInt(k, 10);
                        if (isNaN(idx) || idx < 0 || idx >= N_LAYERS) continue;
                        if (typeof v.attn === "number") {
                            strengths[idx].attn = v.attn;
                            if (v.attn > 0.001) lastAttn[idx] = v.attn;
                        }
                        if (typeof v.ff === "number") {
                            strengths[idx].ff = v.ff;
                            if (v.ff > 0.001) lastFf[idx] = v.ff;
                        }
                        activeLayers.add(idx);
                    }
                    node.setDirtyCanvas(true, true);
                } catch(e) { /* malformed JSON — ignore */ }
            }

            // Use defineProperty to catch ComfyUI's direct .value = writes (linked inputs)
            setTimeout(() => {
                const lsw = W("layer_strengths");
                if (lsw) {
                    let _val = lsw.value ?? "{}";
                    const _origCb = lsw.callback?.bind(lsw);
                    Object.defineProperty(lsw, "value", {
                        get() { return _val; },
                        set(v) {
                            _val = v;
                            _origCb?.(v);
                            applyStrengthJSON(v);
                        },
                        configurable: true,
                    });
                }
            }, 10);
            // ─────────────────────────────────────────────────────────────────

            let dragLayer    = -1;
            let dragComp     = null;   // "attn" | "ff"
            let shiftDrag    = false;
            let dragStartY   = 0;
            let dragStartVal = 0;

            // Hit-test bounds
            let _graphBounds = null;
            let _btnBounds   = {};
            let _colWidth    = 0;

            function syncWidget() {
                const w = W("layer_strengths");
                if (!w) return;
                // Only emit layers that differ from 1.0/1.0 or are explicitly set
                const out = {};
                for (let i = 0; i < N_LAYERS; i++) {
                    out[i] = { attn: +strengths[i].attn.toFixed(4), ff: +strengths[i].ff.toFixed(4) };
                }
                const v = JSON.stringify(out);
                w.value = v;
                w.callback?.(v);
            }

            function globalStrength() {
                return W("strength_model")?.value ?? 1.0;
            }

            function layerFromX(mx, gX, gW) {
                const col = (mx - gX) / gW;
                return clamp(Math.floor(col * N_LAYERS), 0, N_LAYERS - 1);
            }

            function compFromY(my, gY, gH) {
                // Top half = attn, bottom half = ff
                return (my - gY) < gH / 2 ? "attn" : "ff";
            }

            function yToStrength(my, gY, gH) {
                // Bar grows upward from bottom of panel
                const norm = clamp(1.0 - (my - gY) / gH, 0, 1);
                return +(norm * STR_MAX).toFixed(3);
            }

            // ── Widget ────────────────────────────────────────────────────────
            const gw = {
                name: "zimage_lora_graph",
                type: "zimage_lora_graph",
                computeSize(width) { return [width, WIDGET_H]; },

                draw(ctx, node, width, y) {
                    const iW      = width - PAD * 2;
                    const gs      = globalStrength();
                    const colW    = iW / N_LAYERS;
                    _colWidth     = colW;

                    // ── Button row ────────────────────────────────────────────
                    const bY   = y + PAD;
                    const bH   = BTN_ROW_H - 4;
                    const btns = [
                        { key: "reset",   label: "↺ Reset All",       x: PAD },
                        { key: "mirror",  label: "⇄ Attn→FF",         x: PAD + iW * 0.34 },
                        { key: "flatten", label: "▬ Flatten to global", x: PAD + iW * 0.62 },
                    ];
                    const btnW = iW * 0.3;
                    btns.forEach(btn => {
                        btn.w = btnW; btn.y = bY; btn.h = bH;
                        roundRect(ctx, btn.x, btn.y, btn.w, btn.h, 3);
                        ctx.fillStyle   = "#1a1a2e";
                        ctx.fill();
                        ctx.strokeStyle = "#3a3a5a"; ctx.lineWidth = 1; ctx.stroke();
                        ctx.fillStyle   = "#6655aa"; ctx.font = "bold 9px monospace";
                        ctx.textAlign   = "center";
                        ctx.fillText(btn.label, btn.x + btn.w / 2, btn.y + btn.h * 0.68);
                        _btnBounds[btn.key] = btn;
                    });
                    ctx.textAlign = "left";

                    // ── Graph area ────────────────────────────────────────────
                    const gX = PAD, gY = bY + bH + 6, gW = iW, gH = GRAPH_H;
                    _graphBounds = { x: gX, y: gY, w: gW, h: gH };

                    ctx.fillStyle   = "#0a0a18";
                    ctx.strokeStyle = "#2e2e4a"; ctx.lineWidth = 1;
                    roundRect(ctx, gX, gY, gW, gH, 5); ctx.fill(); ctx.stroke();

                    // Split line between attn and ff halves
                    const midY = gY + gH / 2;
                    ctx.strokeStyle = "#1c1c30"; ctx.lineWidth = 0.5;
                    ctx.setLineDash([3, 4]);
                    ctx.beginPath(); ctx.moveTo(gX + 1, midY); ctx.lineTo(gX + gW - 1, midY); ctx.stroke();
                    ctx.setLineDash([]);

                    // Global strength reference line
                    const refNorm  = clamp(gs / STR_MAX, 0, 1);
                    const refYAttn = gY        + (gH / 2) * (1 - refNorm);
                    const refYFf   = gY + gH/2 + (gH / 2) * (1 - refNorm);
                    [refYAttn, refYFf].forEach(ry => {
                        ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 1;
                        ctx.setLineDash([4, 4]);
                        ctx.beginPath(); ctx.moveTo(gX + 1, ry); ctx.lineTo(gX + gW - 1, ry); ctx.stroke();
                        ctx.setLineDash([]);
                    });

                    // Bars
                    for (let i = 0; i < N_LAYERS; i++) {
                        const { attn, ff } = strengths[i];
                        const isActive  = activeLayers.has(i);
                        const barX      = gX + i * colW;
                        const barInnerW = Math.max(1, colW - 2);
                        const innerX    = barX + 1;

                        // ── Attn bar (top half) ───────────────────────────────
                        const attnNorm = clamp(attn / STR_MAX, 0, 1);
                        const attnBarH = (gH / 2 - 2) * attnNorm;
                        const attnBarY = gY + (gH / 2 - 2) - attnBarH;

                        if (isActive && attn > 0.001) {
                            const ag = ctx.createLinearGradient(0, attnBarY, 0, attnBarY + attnBarH);
                            ag.addColorStop(0, dragLayer === i && dragComp === "attn" ? "#d0b8ff" : "#9b7fff");
                            ag.addColorStop(1, "#3a1a6a");
                            ctx.fillStyle = ag;
                        } else {
                            ctx.fillStyle = isActive ? "#1a1a2e" : "#111120";
                        }
                        ctx.fillRect(innerX, attnBarY, barInnerW, attnBarH);

                        // Attn top cap
                        if (isActive && attn > 0.001) {
                            ctx.fillStyle = dragLayer === i && dragComp === "attn" ? "#ffffff" : "#c8b8ff";
                            ctx.fillRect(innerX, attnBarY, barInnerW, 2);
                        }

                        // ── FF bar (bottom half) ──────────────────────────────
                        const ffNorm  = clamp(ff / STR_MAX, 0, 1);
                        const ffBarH  = (gH / 2 - 2) * ffNorm;
                        const ffBarY  = gY + gH / 2 + 2;

                        if (isActive && ff > 0.001) {
                            const fg = ctx.createLinearGradient(0, ffBarY, 0, ffBarY + ffBarH);
                            fg.addColorStop(0, dragLayer === i && dragComp === "ff" ? "#a0f0ff" : "#50c8f0");
                            fg.addColorStop(1, "#1a3a40");
                            ctx.fillStyle = fg;
                        } else {
                            ctx.fillStyle = isActive ? "#1a1a2e" : "#111120";
                        }
                        ctx.fillRect(innerX, ffBarY, barInnerW, ffBarH);

                        // FF top cap
                        if (isActive && ff > 0.001) {
                            ctx.fillStyle = dragLayer === i && dragComp === "ff" ? "#ffffff" : "#a0e8ff";
                            ctx.fillRect(innerX, ffBarY, barInnerW, 2);
                        }

                        // Hover highlight outline
                        if (dragLayer === i) {
                            ctx.strokeStyle = "rgba(255,255,255,0.3)";
                            ctx.lineWidth = 1;
                            ctx.strokeRect(innerX, gY + 1, barInnerW, gH - 2);
                        }

                        // Layer index label (every 5)
                        if (i % 5 === 0) {
                            ctx.fillStyle = "#3a3a5a"; ctx.font = "7px monospace";
                            ctx.textAlign = "center";
                            ctx.fillText(String(i), barX + colW / 2, gY + gH - 3);
                            ctx.textAlign = "left";
                        }
                    }

                    // Panel labels
                    ctx.fillStyle = "#6644aa"; ctx.font = "bold 8px monospace";
                    ctx.fillText("ATTN", gX + 3, gY + 11);
                    ctx.fillStyle = "#2a7a9a"; ctx.font = "bold 8px monospace";
                    ctx.fillText("FF",   gX + 3, gY + gH / 2 + 11);

                    // Value tooltip for dragged bar
                    if (dragLayer >= 0) {
                        const val = dragComp === "attn" ? strengths[dragLayer].attn : strengths[dragLayer].ff;
                        const tx  = clamp(gX + (dragLayer + 0.5) * colW, gX + 20, gX + gW - 20);
                        ctx.fillStyle = "#0d0d1a";
                        roundRect(ctx, tx - 20, gY + gH / 2 - 10, 40, 14, 3);
                        ctx.fill();
                        ctx.fillStyle = dragComp === "attn" ? "#c8b8ff" : "#a0e8ff";
                        ctx.font = "bold 9px monospace"; ctx.textAlign = "center";
                        ctx.fillText(val.toFixed(2), tx, gY + gH / 2);
                        ctx.textAlign = "left";
                    }

                    // ── Label row ─────────────────────────────────────────────
                    const lY = gY + gH + 6;
                    ctx.fillStyle = "#4a4a6a"; ctx.font = "8px monospace";
                    ctx.fillText(`global: ${gs.toFixed(2)}`, gX + 2, lY + 11);
                    ctx.fillStyle = "#6644aa";
                    ctx.fillText("■ attn", gX + 60, lY + 11);
                    ctx.fillStyle = "#2a7a9a";
                    ctx.fillText("■ ff", gX + 100, lY + 11);
                    ctx.fillStyle = "#3a3a5a";
                    ctx.fillText("drag ↕ | click=toggle | shift-drag=all", gX + gW - 195, lY + 11);
                },

                mouse(event, pos, node) {
                    const [mx, my] = pos;
                    const gs = globalStrength();

                    // ── Buttons ───────────────────────────────────────────────
                    if (event.type === "pointerdown") {
                        for (const [key, b] of Object.entries(_btnBounds)) {
                            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                                if (key === "reset") {
                                    strengths = defaultStrengths();
                                } else if (key === "mirror") {
                                    for (let i = 0; i < N_LAYERS; i++) strengths[i].ff = strengths[i].attn;
                                } else if (key === "flatten") {
                                    for (let i = 0; i < N_LAYERS; i++) {
                                        strengths[i].attn = gs;
                                        strengths[i].ff   = gs;
                                    }
                                }
                                syncWidget();
                                node.setDirtyCanvas(true, true);
                                return true;
                            }
                        }
                    }

                    // ── Graph interactions ────────────────────────────────────
                    if (!_graphBounds) return false;
                    const { x: gX, y: gY, w: gW, h: gH } = _graphBounds;
                    const inGraph = mx >= gX && mx <= gX + gW && my >= gY && my <= gY + gH;

                    if (event.type === "pointerdown" && inGraph) {
                        const li   = layerFromX(mx, gX, gW);
                        const comp = compFromY(my, gY, gH);
                        shiftDrag    = event.shiftKey;
                        dragLayer    = li;
                        dragComp     = comp;
                        dragStartY   = my;
                        dragStartVal = strengths[li][comp];

                        // Simple click = toggle
                        if (!shiftDrag) {
                            const cur = strengths[li][comp];
                            if (cur < 0.01) {
                                strengths[li][comp] = comp === "attn" ? lastAttn[li] : lastFf[li];
                            }
                            // actual drag handled on pointermove
                        }
                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    if (event.type === "pointerup" || event.type === "pointercancel") {
                        // If barely moved it's a toggle
                        if (dragLayer >= 0 && Math.abs(my - dragStartY) < 4) {
                            const comp = dragComp;
                            const cur  = strengths[dragLayer][comp];
                            if (cur < 0.01) {
                                strengths[dragLayer][comp] = comp === "attn" ? lastAttn[dragLayer] : lastFf[dragLayer];
                            } else {
                                if (comp === "attn") lastAttn[dragLayer] = cur;
                                else                  lastFf[dragLayer]   = cur;
                                strengths[dragLayer][comp] = 0;
                            }
                        }
                        dragLayer = -1; dragComp = null;
                        syncWidget();
                        node.setDirtyCanvas(true, true);
                        return false;
                    }

                    if (event.type === "pointermove" && dragLayer >= 0 && inGraph) {
                        const dy    = dragStartY - my;
                        const delta = (dy / (gH / 2)) * STR_MAX;
                        const newVal = clamp(dragStartVal + delta, STR_MIN, STR_MAX);

                        if (shiftDrag) {
                            // Move all active layers together
                            const diff = newVal - dragStartVal;
                            for (let i = 0; i < N_LAYERS; i++) {
                                if (activeLayers.has(i)) {
                                    strengths[i][dragComp] = clamp(strengths[i][dragComp] + diff, STR_MIN, STR_MAX);
                                }
                            }
                            dragStartVal = newVal;
                            dragStartY   = my;
                        } else {
                            strengths[dragLayer][dragComp] = newVal;
                        }

                        syncWidget();
                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    return false;
                },

                serializeValue() { return undefined; },
            };

            if (!node.widgets) node.widgets = [];
            node.widgets.push(gw);
            node.setSize(node.computeSize());

            // ── Restore state after page refresh ─────────────────────────────
            // onConfigure fires after ComfyUI rehydrates all widget values from
            // the saved workflow. At that point layer_strengths already has its
            // saved JSON — we just need to parse it back into `strengths`.
            const origConfigure = node.onConfigure?.bind(node);
            node.onConfigure = function(config) {
                origConfigure?.(config);

                // Give the hidden widget one tick to receive its saved value
                setTimeout(() => {
                    const w = W("layer_strengths");
                    if (!w || !w.value || w.value === "{}") return;
                    try {
                        const raw = JSON.parse(w.value);
                        if (typeof raw !== "object") return;
                        // Merge saved values into strengths, keeping defaults for
                        // any layer not present in the saved data
                        for (const [k, v] of Object.entries(raw)) {
                            const idx = parseInt(k, 10);
                            if (isNaN(idx) || idx < 0 || idx >= N_LAYERS) continue;
                            if (typeof v.attn === "number") strengths[idx].attn = v.attn;
                            if (typeof v.ff   === "number") strengths[idx].ff   = v.ff;
                            // Restore last-non-zero tracker so toggles work correctly
                            if (v.attn > 0.001) lastAttn[idx] = v.attn;
                            if (v.ff   > 0.001) lastFf[idx]   = v.ff;
                        }
                        node.setDirtyCanvas(true, true);
                    } catch (e) {
                        // Malformed JSON — stay with defaults, no crash
                    }
                }, 0);
            };

            // ── Hide/show lora_name widget based on override link ─────────────
            function updateLoraNameVisibility() {
                const overrideInput = node.inputs?.find(inp => inp.name === "lora_name_override");
                const isLinked = overrideInput?.link != null;
                const loraWidget = node.widgets?.find(w => w.name === "lora_name");
                if (!loraWidget) return;
                if (isLinked) {
                    loraWidget._origType         = loraWidget._origType || loraWidget.type;
                    loraWidget._origComputeSize  = loraWidget._origComputeSize || loraWidget.computeSize;
                    loraWidget.type              = "hidden_lora_override";
                    loraWidget.computeSize       = () => [0, -4];
                } else {
                    if (loraWidget._origType) {
                        loraWidget.type        = loraWidget._origType;
                        loraWidget.computeSize = loraWidget._origComputeSize;
                        delete loraWidget._origType;
                        delete loraWidget._origComputeSize;
                    }
                }
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }

            node.onConnectionsChange = function(type, index, connected, link_info) {
                updateLoraNameVisibility();
            };

            // Run once on creation in case already linked (workflow reload)
            setTimeout(() => updateLoraNameVisibility(), 50);
            // ─────────────────────────────────────────────────────────────────

            return result;
        };
    },
});
