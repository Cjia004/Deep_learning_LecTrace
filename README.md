# EEG Tokenization — LecTrace presentation

本目录是一份独立演示副本，不会改动原始项目。英文正文、图表和变量说明配有中文讲解提示，按约 10 分钟设计，包含 5 个关键执行状态。只加载现有模型推理，不执行训练、重新拟合标准化器或探针。

## 先启动

在 VS Code 打开本文件所在目录，在终端激活原项目的 Python 环境（已确认 dlcj 可执行推理）：

```powershell
conda activate dlcj
python -m pip install lectrace
lectrace serve 01_eeg_tokenization.py
```

如果终端没有 conda 命令，可使用解释器完整路径：

```powershell
& "D:\anaconda\envs\dlcj\python.exe" -m pip install lectrace
& "D:\anaconda\envs\dlcj\Scripts\lectrace.exe" serve 01_eeg_tokenization.py
```

运行以上命令时，当前目录必须是解压后的 lectrace_eeg 目录。打开终端给出的地址（官方默认 http://localhost:7000）。用右箭头前进，左箭头后退；Shift+右箭头跳过子调用。脚本中的 @stepover 已避免展开神经网络内部的大量细节。

**当前验证状态：**已在 Python 3.11.16、PyTorch 2.14.0+cu132、NumPy 2.4.6 上完成 CPU 推理测试。五组中间结果与原始 forward 完全一致，最大绝对误差均为 0。当前制作环境无法解析 PyPI 下载地址，因此没有安装 LecTrace，也没有完成实际网页渲染测试。请在自己的终端完成安装并启动；不应把现有验证记录误称为浏览器验收。官方说明：https://github.com/praisegee/lectrace

可先仅检查真实模型推理：
```powershell
python _verify.py
```
该检查只将讲义中的 text/note/plot 替换为内容收集器，真实模型、数据、张量运算和原 forward 对照断言均照常运行。它不验证 LecTrace 的追踪和渲染。

## 演示路线

| 时间 | 内容 | 应解释的变化 |
|---|---|---|
| 0:00–2:30 | 研究问题、医疗研究背景、数据与评价 | tokenization 是研究主体，BDI 是下游分组任务 |
| 2:30–3:15 | State 1: 标准化输入 | 2 个真实测试窗口，4 通道 × 512 采样点 |
| 3:15–4:15 | State 2: patch embedding | 每通道 7 个重叠 patch，128 维波形映射到 64 维 |
| 4:15–5:30 | State 3: 时间分支 | 通道内时间建模，再跨通道平均，得到 7 个 token |
| 5:30–6:45 | State 4: 空间分支 | 时间片内通道建模，再跨时间平均，得到 4 个 token |
| 6:45–8:00 | State 5: 融合与 CLS 分类 | 12 个 token → CLS → logit → 睁眼得分 |
| 8:00–9:15 | 回到完整实验 | BDI、阳性对照与探针的评价单位和结论不同 |
| 9:15–10:00 | 总结与限制 | 当前证据、SSL 实现问题、独立数据验证 |

## 示例是什么

- 使用冻结 ZIP 中原样复制的 Step 10d ch4 TS、fold 0、seed 0 checkpoint（selected epoch 32）。
- 原始 src/model.py 和 src/config.py 原样复制，并记录 SHA-256。
- 测试被试 sub-010，run-01，窗口 0（闭眼）和 32（睁眼）。
- 选取规则：按名称排序，取 fold 0 中首个同时有两种条件的被试/录音，再分别取首个通过 ch4 幅值质控的窗口；选择先于预测，不筛选正确案例。
- 两个窗口对应的睁眼得分约为 0.203、0.427，阈值为 >0.5。因此闭眼窗口预测正确，睁眼窗口预测错误。这是刻意保留的真实结果，不应改成伪造的“完美示例”。
- 一例错误不推翻总体 AUC；两例预测也不能用于估计总体性能。当前只是一个 checkpoint，报告控制任务使用 3 个种子的折外概率平均。
- 演示末尾 ch16 S 的结果来自保存汇总表，明确不同于演示使用的 ch4 TS。
- 输入样例已是原项目处理后的微伏窗口，演示只重现标准化和模型推理，不重新滤波或分窗。

## 文件与复现

- 01_eeg_tokenization.py：提交用 LecTrace 主讲义。
- _runtime.py：数据加载、可视化和原 forward 对照助手。
- src/：从冻结 ZIP 原样复制的最小模型依赖。
- assets/：原 checkpoint、两个 EEG 窗口、折表、三份原结果表、溯源清单。
- assets/manifest.json：来源、选取规则、结果和文件哈希。
- verification.json：本机真实推理测试结果。
- speaker_notes_zh.md：中文讲解要点。
- preview.png：真实张量的静态核对图，仅是内容预览，不是 LecTrace 网页截图。

需要提交 Python 源码，因此不要只提交截图或网页。无需把整个原始 EEG 数据集复制进此目录。该目录可以与最终报告及补充材料一起打包。代码没有发布到任何外部网站。

## 结论边界

BDI 配对 AUC 差值区间包含 0；不称模型优于 LR，不称模型诊断抑郁症。探针来自 Step 10d 控制任务，支持线性可解码性，不证明因果决策依赖。热图显示 token 激活而非注意力矩阵，不赋予特征维度固定生理意义。总体指标不由两个示例重新估计。

## 来源与 AI 披露

- 数据集：[EEG: Depression rest, ds003478 v1.1.0](https://doi.org/10.18112/openneuro.ds003478.v1.1.0)
- Transformer：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 演示框架：[LecTrace](https://github.com/praisegee/lectrace)，安装来自 PyPI 官方包。
- 模型与全部实验数值：用户冻结的 final project.zip；SHA-256 记录在 manifest 中。
- 本演示为 AI 辅助生成，并进行了真实推理与源码一致性核查。作者仍需亲自运行网页、熟悉五个状态并核对讲稿。请在最终提交的 AI 使用说明中补充 LecTrace 脚本的 AI 辅助情况。

