"""Read-only inference helpers, excluded from the lecture trace."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from src.model import STDualTokenizer

HERE = Path(__file__).resolve().parent

def load_demo():
    manifest = json.loads((HERE / "assets/manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["file_sha256"].items():
        actual = hashlib.sha256((HERE / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Bundled input changed: {relative}")
    # Only load this hash-verified, trusted project checkpoint.
    ckpt = torch.load(HERE / "assets/ch4_TS_fold0_seed0.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = STDualTokenizer(manifest["channels"], mode="TS", n_samples=512)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    with np.load(HERE / "assets/example_windows.npz", allow_pickle=False) as d:
        raw = torch.from_numpy(d["raw_uv"].copy())
    mean = torch.as_tensor(ckpt["mean"], dtype=torch.float32)[None, :, None]
    std = torch.as_tensor(ckpt["std"], dtype=torch.float32)[None, :, None]
    assert torch.all(std > 0)
    return model, raw, mean, std, manifest

def shape(tensor):
    return list(tensor.shape)

def waveform(x, channels):
    a = x[0].detach().cpu().numpy()
    values = [{"Time (s)": i / 128, "Value": float(v), "Channel": channels[c]}
              for c in range(len(channels)) for i, v in enumerate(a[c])]
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "First held-out window: standardized EEG (eyes closed)",
            "data": {"values": values}, "facet": {"row": {"field": "Channel", "type": "nominal", "sort": channels}},
            "spec": {"width": 560, "height": 55, "mark": {"type": "line", "color": "#2563a6"},
                     "encoding": {"x": {"field": "Time (s)", "type": "quantitative"},
                                  "y": {"field": "Value", "type": "quantitative", "title": "z score"}}}}

def heatmap(x, labels, title):
    a = x[0].detach().cpu().numpy()
    values = [{"Token": labels[i], "Feature": j, "Activation": float(a[i, j])}
              for i in range(a.shape[0]) for j in range(a.shape[1])]
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "title": title,
            "width": 560, "height": max(110, len(labels) * 23), "data": {"values": values}, "mark": "rect",
            "encoding": {"x": {"field": "Feature", "type": "ordinal", "axis": {"values": [0, 16, 32, 48, 63], "labelAngle": 0}},
                         "y": {"field": "Token", "type": "ordinal", "sort": labels},
                         "color": {"field": "Activation", "type": "quantitative", "scale": {"scheme": "redblue", "domainMid": 0}},
                         "tooltip": [{"field": "Token"}, {"field": "Feature"}, {"field": "Activation", "format": ".3f"}]}}

def verify_forward(model, x, embeddings, temporal, spatial, fused, logits):
    with torch.inference_mode():
        ref_logits, parts = model(x, return_tokens=True)
    errors = {}
    for name, a, b in [("patch", embeddings, parts["patch"]), ("temporal", temporal, parts["temporal"]),
                       ("spatial", spatial, parts["spatial"]), ("fused", fused, parts["fused"]), ("logit", logits, ref_logits)]:
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        errors[name] = float((a - b).abs().max())
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    return errors

def evidence_table(meta):
    r = meta["results"]
    rows = [
        "| Evidence | Saved result | Interpretation |",
        "|---|---|---|",
        f"| BDI: ch4 TS vs LR | Subject OOF AUC {r['bdi_auc']:.3f} vs {r['bdi_lr_auc']:.3f}; paired difference {r['bdi_delta']:.3f}, 95% CI [{r['bdi_delta_lo']:.3f}, {r['bdi_delta_hi']:.3f}] | No established advantage over LR |",
        f"| Control: ch16 S | Window OOF AUC {r['control_auc']:.3f}, 95% CI [{r['control_lo']:.3f}, {r['control_hi']:.3f}] | Recording conditions are distinguishable |",
        f"| Probe: ch16 S, spatial, alpha | Held-out R² {r['probe_trained']:.3f} ± {r['probe_trained_sd']:.3f} vs random {r['probe_random']:.3f} ± {r['probe_random_sd']:.3f} (mean ± SD) | Band-power information is linearly decodable |",
    ]
    return "\n".join(rows)

