from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT /"dataset"
PROC_DIR = ROOT /"processed"
LOG_DIR = ROOT /"logs"
RESULT_DIR = ROOT /"results"
FS_TARGET = 128
BAND = (0.5, 45.0)
WIN_S, STEP_S, TRIM_S = 4.0, 2.0, 2.0
AMP_UV = 150.0
BDI_CONTROL_MAX = 7
BDI_HIGH_MIN = 13
NON_SCALP = ["M1", "M2", "CB1", "CB2", "HEOG", "VEOG", "EKG"]
SUBJECT_TABLE = PROC_DIR / "subject_table.csv"
SEX_MAP = {1: "female", 2: "male"}
MASTOIDS = ["M1", "M2"]
FIG_DIR = RESULT_DIR / "figures"
# ---------- 伪影扫描 ----------
GROSS_ART_UV = 1000.0
SCAN_CSV = PROC_DIR / "gross_artifact_scan.csv"
# ---------- 事件 -> 块 ----------
MIN_BLOCK_S = 20.0      # 短于此的"块"视为游离标记, 丢弃
GAP_S = 5.0             # 同码标记间隔超过此值则断开
CODE2COND = {1: "closed", 3: "closed", 5: "closed",
             2: "open", 4: "open", 6: "open"}   # 奇数=闭眼, 偶数=睁眼(已在有标签文件上验证)
BLOCK_SCAN_CSV = PROC_DIR / "block_artifact_scan.csv"
N_CH_EXPECTED = 62
REF_CSV = PROC_DIR / "reference_comparison.csv"
# ---------- 通道顺序(来源: Step 3e, 241 个录音完全一致) ----------
SCALP_60 = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
            "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
            "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4",
            "PO6", "PO8", "O1", "OZ", "O2"]

# ---------- 批量预处理输出 ----------
WIN_DIR = PROC_DIR / "windows"
PREPROC_CSV = PROC_DIR / "preprocess_summary.csv"
COND2INT = {"closed": 0, "open": 1}
CH_STD_CSV = PROC_DIR / "channel_std.csv"
# ---------- 通道配置(Step 7 起使用) ----------
CH_4 = ["FP1", "FP2", "F3", "F4"]
CH_16 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "C4", "T8", "P3", "PZ", "P4", "O1", "O2"]
CHANNEL_SETS = {"ch4": CH_4, "ch16": CH_16, "ch60": SCALP_60}
# ---------- 剔窗与交叉验证(Step 7/8 的决定) ----------
REJECT_TH_UV = 150.0     # 窗内所选通道任一峰值超过则剔除; 在看分类结果之前决定, 之后不再调整
N_FOLDS = 5
CV_SEED = 0
FOLDS_CSV = PROC_DIR / "folds.csv"
# ---------- 频带(基线与之后的频谱 tokenizer 共用) ----------
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
# ---------- 模型默认值(Step 10a; 超参数不在 BDI 上调) ----------
MODEL = dict(d=64, heads=4, ffn_mult=4, layers_branch=1, layers_fuse=1, patch_len=128, patch_stride=64, dropout=0.1)
# ---------- Step 10b: EEG Tokenizer 训练配置 ----------
TRAIN = dict(
    seeds=[0, 1, 2, 3, 4],
    max_epochs=400,      # 允许后续在 300–500 范围内调整
    batch_size=128,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=40,
    min_delta=1e-4,
    val_fraction=0.20,   # 从外层训练被试中切出 20% 作验证
    num_workers=0,       # Windows 环境先使用 0，确认稳定后再调高
    grad_clip=1.0,
    ssl_mask_ratio=0.40,
)