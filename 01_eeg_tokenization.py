"""Ten-minute LecTrace presentation: five selected inference states.
AI-assisted presentation; inference only, no training.
Operations follow the frozen src/model.py, STDualTokenizer.forward.
"""
import torch
from lectrace import text, note, plot
from _runtime import load_demo, shape, waveform, heatmap, verify_forward, evidence_table

def main():
    model, raw_uv, mean, std, meta = load_demo()  # @hide
    text("# EEG Tokenization: Temporal and Spatial Representations")
    text("**Question:** how should a neural network organize EEG across time and channels? We evaluate continuous token representations using BDI grouping, a recording-condition control, and band-power probes.")
    text("**Healthcare context:** EEG may support research on depressive symptoms. Our downstream target is questionnaire-based BDI grouping. This project does not establish clinical diagnosis or a depression biomarker.")
    text("**Data and evaluation:** OpenNeuro ds003478 v1.1.0; 121 included participants. Subject-disjoint outer folds and training-derived normalization. All overlapping windows from one participant remain in the same fold.")
    note("0:00–2:30｜介绍研究对象是tokenization，BDI只是下游评价。大量重叠窗口不等于大量独立被试。先讲研究问题，再用一个真实推理过程解释设计。")
    text("**Walkthrough:** frozen Step 10d ch4 TS checkpoint, outer fold 0, seed 0. Two windows from one held-out participant: the first retained closed-eye and open-eye window in recording order. Selection does not depend on predictions. This single checkpoint illustrates computation; reported results use multi-seed predictions.")
    text("## State 1 — Apply training-fold normalization")
    x = (raw_uv - mean) / std
    input_shape = shape(x)  # @inspect input_shape
    text("[batch, channels, samples] = [2, 4, 512]: two 4-second windows at 128 Hz. Channels: FP1, FP2, F3, F4. Saved training means and standard deviations are applied without fitting to these test windows.")
    plot(waveform(x, meta["channels"]))
    note("2:30–3:15｜波形来自真实留出被试。两个窗口只展示计算，不能用于估计AUC。标准化使用checkpoint中内层训练集的统计量。")
    text("## State 2 — Embed overlapping waveform patches")
    patches = x.unfold(-1, model.L, model.stride)
    embeddings = model.patch(patches)  # @stepover
    patch_shape, embedding_shape = shape(patches), shape(embeddings)  # @inspect patch_shape embedding_shape
    first_patch_features = embeddings[0, 0, 0, :8].tolist()  # @inspect first_patch_features
    text("A 128-sample patch moves by 64 samples: 1-second patches with 0.5-second stride. Each channel yields 7 patches. Shared Linear(128, 64) + LayerNorm produces continuous 64-dimensional embeddings, not discrete codebook IDs.")
    note("3:15–4:15｜[2,4,512] → [2,4,7,128] → [2,4,7,64]。7=(512−128)/64+1。右侧数值来自实际投影，不把某个特征维度解释成特定频带。")
    text("## State 3 — Attend over time, then average channels")
    temporal_input = (embeddings + model.t_pos).reshape(2 * 4, 7, 64)
    temporal_encoded = model.t_enc(temporal_input)  # @stepover
    temporal = temporal_encoded.reshape(2, 4, 7, 64).mean(dim=1)
    temporal_input_shape, temporal_shape = shape(temporal_input), shape(temporal)  # @inspect temporal_input_shape temporal_shape
    plot(heatmap(temporal, [f"T{i}" for i in range(7)], "Temporal tokens: first held-out window"))
    text("Attention acts on 7 time patches separately within each channel. Channel averaging leaves 7 temporal tokens per window. The heatmap shows feature activations, not attention weights or physiological labels.")
    note("4:15–5:30｜每个通道对应一条长度7的序列，共batch×4条。先建模通道内时间关系，再平均4个通道。每行是一个时间位置的表示。")
    text("## State 4 — Attend over channels, then average time")
    spatial_input = embeddings + model.ch_emb(model.ch_idx)[None, :, None, :]
    spatial_input = spatial_input.permute(0, 2, 1, 3).reshape(2 * 7, 4, 64)
    spatial_encoded = model.s_enc(spatial_input)  # @stepover
    spatial = spatial_encoded.reshape(2, 7, 4, 64).mean(dim=1)
    spatial_input_shape, spatial_shape = shape(spatial_input), shape(spatial)  # @inspect spatial_input_shape spatial_shape
    plot(heatmap(spatial, meta["channels"], "Spatial tokens: first held-out window"))
    text("Each time patch supplies a sequence of 4 channels. Channel embeddings identify electrodes. Attention acts across channels; averaging the 7 patches leaves 4 spatial tokens. This supervised spatial path does not add temporal positional embeddings.")
    note("5:30–6:45｜两分支共用patch embeddings，但注意力序列轴不同。S在同一时间片比较通道，再汇总时间。不能根据两幅热图的颜色判断分支优劣。")
    text("## State 5 — Fuse tokens and classify from CLS")
    fusion_input = torch.cat([model.cls.expand(2, -1, -1), temporal + model.type_emb[0], spatial + model.type_emb[1]], dim=1)
    fused = model.fuse(fusion_input)  # @stepover
    logits = model.head(fused[:, 0]).squeeze(-1)  # @stepover
    p_open = torch.sigmoid(logits)
    fusion_shape, probabilities = shape(fusion_input), p_open.tolist()  # @inspect fusion_shape probabilities
    predicted_conditions = ["open" if p > 0.5 else "closed" for p in probabilities]  # @inspect predicted_conditions
    text("Fusion receives 1 CLS + 7 temporal + 4 spatial = 12 tokens, each with 64 features. The head reads the fused CLS vector; sigmoid yields the score for eyes open. True conditions, in order: **closed, open**. These are not BDI probabilities or calibrated clinical risks.")
    forward_errors = verify_forward(model, x, embeddings, temporal, spatial, fused, logits)  # @hide
    max_forward_error = max(forward_errors.values())  # @inspect max_forward_error
    text("Every displayed intermediate representation and final logit is checked against the original STDualTokenizer.forward. The assertion must pass before the lecture can complete.")
    note("6:45–8:00｜类型嵌入区分T和S，分类头读取CLS。显示真实输出，包括可能的错误，不按正确率筛选示例。单个checkpoint的示例预测不能替代全体折外结果。")
    text("## What the full experiments support")
    text(evidence_table(meta))
    text("BDI: five-seed probabilities averaged per subject. Control: three-seed probabilities averaged per window, with subject-cluster intervals. Probe: mean ± SD across folds and seeds, not confidence intervals; control-task checkpoints, not BDI or SSL checkpoints.")
    note("8:00–9:15｜BDI配对区间包含0；对照说明可区分记录条件；探针说明可线性读取频带信息，不说明分类器实际使用该信息。表中ch16 S与本次演示ch4 TS不是同一配置。")
    text("## Conclusion")
    text("The tokenizer learns useful recording-condition representations, but current BDI evaluation does not establish an advantage over band-power LR. Fusion and more channels do not give consistent gains. Current SSL results are limited by the masking/reconstruction implementation. Next: improve the visible-information path, reduce overlap shortcuts, and evaluate larger independent datasets.")
    text("**AI disclosure:** AI assisted code generation for downstream validation and probes, report outlining, and this LecTrace presentation. The author is responsible for understanding and validating the submitted work.")
    text("**Sources:** [Dataset](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [LecTrace](https://github.com/praisegee/lectrace). Model operations and results come from the frozen project; see the provenance manifest.")
    note("9:15–10:00｜总结贡献为设计、比较和验证。样本量、性别关联、残余伪影及SSL实现限制结论。下一步改进目标并做独立数据验证。本演示不训练模型。")

if __name__ == "__main__":
    main()

