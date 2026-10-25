import torch
import torch.nn as nn

from src import config


class STDualTokenizer(nn.Module):
    """时空双分支 tokenizer + 分类头。
    mode: "T" 只用时间分支, "S" 只用空间分支, "TS" 两者融合。
    输入 x: (B, C, n_samples), 已按训练折统计量标准化的 EEG; channels: 通道名列表(决定通道嵌入的索引)。
    forward(x) 返回 logit (B,); forward(x, return_tokens=True) 额外返回各阶段的 token 字典。"""

    def __init__(self, channels, mode="TS", n_samples=512, **overrides):
        super().__init__()
        if mode not in ("T", "S", "TS"):
            raise ValueError(f"mode must be T, S or TS, got {mode}")
        cfg = {**config.MODEL, **overrides}
        self.mode, self.cfg, self.channels = mode, cfg, list(channels)
        d = cfg["d"]
        self.n_ch = len(self.channels)
        self.L, self.stride = cfg["patch_len"], cfg["patch_stride"]
        self.P = (n_samples - self.L) // self.stride + 1

        self.patch = nn.Sequential(nn.Linear(self.L, d), nn.LayerNorm(d))      # 每个 patch -> d 维
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.t_pos = nn.Parameter(torch.zeros(self.P, d))
        nn.init.trunc_normal_(self.t_pos, std=0.02)

        if "T" in mode:
            self.t_enc = self._encoder(cfg["layers_branch"])
        if "S" in mode:
            self.ch_emb = nn.Embedding(len(config.SCALP_60), d)                # 通道嵌入(按 SCALP_60 的名字索引)
            nn.init.trunc_normal_(self.ch_emb.weight, std=0.02)
            self.register_buffer("ch_idx", torch.tensor([config.SCALP_60.index(c) for c in self.channels]))
            self.s_enc = self._encoder(cfg["layers_branch"])
        if mode == "TS":
            self.type_emb = nn.Parameter(torch.zeros(2, d))                    # 区分时间 token / 空间 token
            nn.init.trunc_normal_(self.type_emb, std=0.02)
        self.fuse = self._encoder(cfg["layers_fuse"])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        # 自监督专用：被遮蔽 patch 的可学习替代向量
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 将编码后的 patch 表征映射回原始 patch 波形
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, self.L),
        )
        # 自监督重建中，学习时间分支与空间分支的融合比例
        if mode == "TS":
            self.recon_fusion_logits = nn.Parameter(
                torch.zeros(2)
            )

    def _encoder(self, n_layers):
        d = self.cfg["d"]
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.cfg["heads"], dim_feedforward=d * self.cfg["ffn_mult"],
            dropout=self.cfg["dropout"], activation="gelu", batch_first=True, norm_first=True)
        return nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d),
                                     enable_nested_tensor=False)
    def _check_input(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape (B, C, T), got {tuple(x.shape)}"
            )

        if x.shape[1] != self.n_ch:
            raise ValueError(
                f"Expected {self.n_ch} channels, got {x.shape[1]}"
            )

        if x.shape[2] < self.L:
            raise ValueError(
                f"Input length {x.shape[2]} is shorter than patch_len {self.L}"
            )

        actual_p = (x.shape[2] - self.L) // self.stride + 1

        if actual_p != self.P:
            raise ValueError(
                f"Expected {self.P} patches from n_samples configuration, "
                f"but input produces {actual_p}. "
                "Check n_samples and input length."
            )

    def forward(self, x, return_tokens=False):
        self._check_input(x)
        B, C, _ = x.shape
        E = self.patch(x.unfold(-1, self.L, self.stride))                      # (B, C, P, d)
        d = E.shape[-1]
        toks, parts = [self.cls.expand(B, -1, -1)], {"patch": E}
        if "T" in self.mode:
            h = (E + self.t_pos).reshape(B * C, self.P, d)                     # 每个通道一条长度 P 的序列
            zt = self.t_enc(h).reshape(B, C, self.P, d).mean(dim=1)            # 对通道取平均 -> (B, P, d)
            parts["temporal"] = zt
            toks.append(zt + self.type_emb[0] if self.mode == "TS" else zt)
        if "S" in self.mode:
            h = E + self.ch_emb(self.ch_idx)[None, :, None, :]
            h = h.permute(0, 2, 1, 3).reshape(B * self.P, C, d)                # 每个 patch 一条长度 C 的序列
            zs = self.s_enc(h).reshape(B, self.P, C, d).mean(dim=1)            # 对时间取平均 -> (B, C, d)
            parts["spatial"] = zs
            toks.append(zs + self.type_emb[1] if self.mode == "TS" else zs)
        tokens = self.fuse(torch.cat(toks, dim=1))                             # (B, 1+P(+C), d)
        logit = self.head(tokens[:, 0]).squeeze(-1)
        if return_tokens:
            parts["fused"] = tokens
            return logit, parts
        return logit

    def token_labels(self, fs=128.0):
        """融合后 token 序列每一行对应什么(用于日志和可视化)。"""
        labels = ["CLS"]
        if "T" in self.mode:
            labels += [f"T{p} [{p * self.stride / fs:.1f}-{(p * self.stride + self.L) / fs:.1f}s]"
                       for p in range(self.P)]
        if "S" in self.mode:
            labels += [f"S:{c}" for c in self.channels]
        return labels
    def forward_masked_reconstruction(self, x, mask_ratio=0.40):
        """
        自监督 masked patch reconstruction。

        参数
        ----
        x : Tensor
            标准化后的 EEG，形状 (B, C, T)。
        mask_ratio : float
            被遮蔽的时间 patch 比例。

        返回
        ----
        loss : Tensor
            仅在被遮蔽 patch 上计算的 MSE 重建损失。

        注意
        ----
        输入标准化统计量必须只从当前内层训练集拟合。
        时间遮蔽位置在同一个样本的所有通道间共享。
        """
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio 必须在 0 和 1 之间")
        self._check_input(x)
        B, C, _ = x.shape

        # 原始 patch：用于生成监督重建目标
        raw = x.unfold(-1, self.L, self.stride)  # (B, C, P, L)
        P = raw.shape[2]

        # 每个样本随机选择一组时间 patch 遮蔽位置，
        # 同一个样本的所有 EEG 通道使用相同的时间遮蔽位置。
        n_mask = max(1, min(P - 1, int(round(P * mask_ratio))))
        noise = torch.rand(B, P, device=x.device)
        mask = torch.zeros(B, P, dtype=torch.bool, device=x.device)
        mask.scatter_(1, noise.argsort(dim=1)[:, :n_mask], True)

        # patch embedding
        E = self.patch(raw)  # (B, C, P, d)

        # 遮蔽输入 patch 的嵌入，而不是把目标波形从重建损失中删除
        E_masked = torch.where(
            mask[:, None, :, None],
            self.mask_token.expand(B, C, P, -1),
            E,
        )

        # 时间分支：每个通道分别沿时间建模
        zt = None
        if "T" in self.mode:
            h = (E_masked + self.t_pos[None, None, :, :])
            h = h.reshape(B * C, P, -1)
            zt = self.t_enc(h).reshape(B, C, P, -1)

        # 空间分支：每个时间 patch 内沿通道建模
        zs = None
        if "S" in self.mode:
            ch = self.ch_emb(self.ch_idx)[None, :, None, :]
            time_pos = self.t_pos[None, None, :, :]

            h = E_masked + ch + time_pos
            h = h.permute(0, 2, 1, 3).reshape(B * P, C, -1)

            zs = self.s_enc(h).reshape(B, P, C, -1)
            zs = zs.permute(0, 2, 1, 3)
        # 按当前模型模式组合每个通道、每个时间 patch 的表征
        if self.mode == "TS":
            fusion_weights = torch.softmax(
                self.recon_fusion_logits, dim=0
            )
            h = fusion_weights[0] * zt + fusion_weights[1] * zs
        elif self.mode == "T":
            h = zt
        else:
            h = zs
        # 重建每个 patch 的标准化波形
        pred = self.recon_head(h)  # (B, C, P, L)

        # 仅在被遮蔽的位置计算损失，避免模型只学习复制可见输入
        # pred 和 raw 的形状均为 (B, C, P, L)
        # mask 的形状为 (B, P)，True 表示该时间 patch 被遮蔽

        sq_error = (pred - raw).pow(2)

        # 仅在被遮蔽的时间 patch 上计算重建误差
        masked_error = (
            sq_error
            * mask[:, None, :, None].to(sq_error.dtype)
        )

        # 对每个样本分别求误差总和
        numerator = masked_error.sum(dim=(1, 2, 3))

        # 每个样本的有效重建元素数量
        denominator = (
            mask.sum(dim=1).to(sq_error.dtype)
            * C
            * self.L
        ).clamp_min(1.0)

        # 返回每个样本的损失，而不是整个 batch 的标量
        loss_per_sample = numerator / denominator

        return loss_per_sample