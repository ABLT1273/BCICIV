# BCICIV 实验目录（BCICIV2a）

基于 [MOABB](https://github.com/NeuroTechX/moabb) 加载 BCI Competition IV 2a 数据集（BNCI2014_001），
实现并对比多种运动想象 EEG 特征提取与分类方法。

---

## 目录结构

```
BCICIV/
└── BCICIV2a/
    ├── framework/           基础设施层（数据、路径、运行时、绘图、注册表）
    ├── models/              模型算法包（特征提取器 + 分类器 + DL 模型适配器）
    │   ├── official_tcn/      TCN 官方 PyTorch 实现
    │   ├── official_atcnet/   ATCNet 官方 braindecode 实现
    │   └── official_drsn/     DRSN-CS 官方 PyTorch 实现
    ├── model_param/         已训练模型参数（.pkl）
    ├── paradigms/           实验范式层（端到端流程编排）
    ├── notebooks/           Jupyter 交互演示
    ├── results/             实验输出（按范式分目录）
    │   ├── within_subject/  被试内 T→E 泛化
    │   │   ├── benchmark_csp_lda/
    │   │   ├── benchmark_fbcsp_dfbcsp/
    │   │   ├── benchmark_nn_models/
    │   │   ├── benchmark_trca_wavelet_cnn/
    │   │   └── dim_reduction_hybrid_fbcsp/
    │   └── loso/            LOSO 跨被试泛化
    │       ├── benchmark_loso/      CSP+LDA, CSP+SVM, FBCSP, EEGNet
    │       └── benchmark_loso_dl/   FBCSP, DeepConvNet, ShallowConvNet, EEGNet
    ├── pre-precess.py       统一实验入口
    ├── test_nn_models_pipeline.py  DL 模型前向验证测试
    ├── IMPLEMENTATION_SUMMARY.md   实现总结
    ├── NN_MODELS_BENCHMARK_README.md  DL 模型基准测试文档
    └── QUICK_REFERENCE.md          快速参考
```

以下层级说明中的相对路径均以 `BCICIV2a/` 作为实验根目录。

---

## 层级说明

### `framework/` — 基础设施

与具体实验解耦，所有范式共用。

| 文件 | 职责 |
|---|---|
| `runtime.py` | 统一配置 MNE / MOABB / matplotlib 缓存目录 |
| `data.py` | 加载单被试 epoch 数据（`load_subject_epochs` / `load_subject_train_test`），按 Session T/E 划分训练/测试集 |
| `cv_split.py` | LOSO / cross-session 切分工具（`generate_loso_folds` / `build_loso_fold`），供 LOSO 范式调用 |
| `constants.py` | 标签映射、通道名、方法排序与显示名等全局常量 |
| `paths.py` | 各目录路径工具函数（`get_model_dir` / `get_results_root` 等） |
| `plotting.py` | 3D UMAP 可视化、对比柱状图、全被试聚合网格图（支持内存直接出图 + 文件拼接两种管路） |
| `registry.py` | 范式注册表，`pre-precess.py` 通过此处动态加载范式模块 |

### `models/` — 模型算法

纯算法实现，不包含实验流程逻辑。每个文件可独立运行（`__main__` 含完整训练/评估流程）。

**经典方法：**

| 文件 | 方法 | 核心类 |
|---|---|---|
| `FBCSP.py` | Filter Bank CSP | `FilterBank` · `OVR_FBCSP_Ensemble` · `NBPWClassifier` · `PairedMIBIF` |
| `DFBCSP.py` | Discriminative FBCSP | `DiscriminativeBandSelector` · `OVR_DFBCSP_Ensemble` |
| `trca_module.py` | Task-Related Component Analysis | `TRCAHybridClassifier` |
| `wavelet_features.py` | Morlet 小波能量特征 | `WaveletEnergyFeatureExtractor` |
| `deep_cnn_features.py` | EEGNet 风格 CNN | `train_tiny_eeg_cnn` · `extract_tiny_eeg_cnn_features` |
| `deep_conv_net.py` | DeepConvNet + ShallowConvNet | `train_deep_conv_net` · `train_shallow_conv_net` (Schirrmeister 2017) |

**深度学习方法（III 期引入）：**

| 文件 | 方法 | 核心类 / 接口 |
|---|---|---|
| `tcn_model.py` | Temporal Convolutional Network | `TCNClassifier` · `train_tcn` · `predict_tcn` · `extract_tcn_features` |
| `atcnet_model.py` | Attention TCN (braindecode) | `ATCNetResult` · `train_atcnet` · `predict_atcnet` · `extract_atcnet_features` |
| `drsn_model.py` | Dilated Residual Shrinkage Network | `DRSNClassifier` · `train_drsn` · `predict_drsn` · `extract_drsn_features` |
| `labram_adapter.py` | LaBraM-Large (TorchEEG) | `LabramAdapter` · `setup_labram_pipeline` · `run_labram_experiment` (Adapter 微调, 冻结预训练权重) |

每个 DL 模型均暴露 `setup_*_pipeline()`（初始化模型）和 `run_*_experiment()`（端到端实验）接口。

**官方实现子目录：**

| 目录 | 内容 |
|---|---|
| `official_tcn/` | TCN 核心模块（`tcn.py` / `conv.py` / `buffer.py` / `pad.py`） |
| `official_atcnet/` | ATCNet braindecode 实现（`atcnet_braindecode.py`） |
| `official_drsn/` | DRSN-CS 实现（`drsn_cs.py`，含 `rsnet18`） |

直接运行模型脚本（以 Subject 1 为例）：

```bash
# 从 BCICIV2a/ 目录执行
../../.venv/bin/python models/FBCSP.py
../../.venv/bin/python models/DFBCSP.py
```

### `model_param/` — 模型参数

存放 `joblib.dump` 保存的已训练流水线，命名规则：`{method}_pretrained_moabb_A{subject:02d}.pkl`。

```python
import joblib
pipeline = joblib.load("model_param/fbcsp_pretrained_moabb_A01.pkl")
y_pred = pipeline['ovr_ensemble'].predict(pipeline['filter_bank'].transform(X_test))
```

### `paradigms/` — 实验范式

将 `models/` 中的算法组装为完整实验流程，通过 `pre-precess.py` 统一调度。
每个范式文件须暴露：

- `build_config_from_namespace(args)` — 从命令行参数构建配置对象
- `run_from_config(config)` — 执行实验并写出结果

| 文件 | 范式键 | 说明 |
|---|---|---|
| `advanced_benchmark.py` | `advanced_feature_benchmark` | 对比 TRCA / Wavelet / CNN 分类性能（含 UMAP 可视化） |
| `hybrid_fbcsp_umap.py` | `hybrid_fbcsp_umap` | C3/C4 + FBCSP 特征融合后 UMAP 降维可视化 |
| `nn_models_benchmark.py` | `nn_models_benchmark` | TCN / ATCNet / DRSN / LaBraM 四模型基准测试 |
| `csp_lda_benchmark.py` | — | CSP+LDA / CSP+SVM within-subject 基准 (9 被试) |
| `fbcsp_dfbcsp_benchmark.py` | — | FBCSP / DFBCSP within-subject 基准 (9 被试) |
| `loso_benchmark.py` | `loso_benchmark` | LOSO: CSP+LDA, CSP+SVM, FBCSP, EEGNet |
| `loso_dl_benchmark.py` | — | LOSO: FBCSP, DeepConvNet, ShallowConvNet, EEGNet (含统一预处理) |
| `cross_session_benchmark.py` | — | Cross-session E→T / T→T 评估 (待运行) |

> **注意**：DL 模型 (TCN / ATCNet / DRSN / EEGNet / DeepConvNet / ShallowConvNet) 均已完成训练回路 (LR scheduler, early stopping, label smoothing)。LaBraM 使用 Adapter 微调 (冻结预训练权重, 仅训练 Adapter + head)。

### `results/` — 实验输出

| 子目录 | 对应范式 | 内容 |
|---|---|---|
| `within_subject/benchmark_trca_wavelet_cnn/` | `advanced_feature_benchmark` | TRCA / Wavelet / CNN 9 被试 CSV + UMAP 3D + summary bar |
| `within_subject/dim_reduction_hybrid_fbcsp/` | `hybrid_fbcsp_umap` | UMAP 嵌入 .npz + 3D 可视化图 |
| `within_subject/benchmark_nn_models/` | `nn_models_benchmark` | TCN / ATCNet / DRSN / LaBraM 9 被试 CSV + summary bar + comparison bar grid |
| `within_subject/benchmark_csp_lda/` | `csp_lda_benchmark.py` | CSP+LDA 9 被试 CSV |
| `within_subject/benchmark_fbcsp_dfbcsp/` | `fbcsp_dfbcsp_benchmark.py` | FBCSP / DFBCSP 9 被试 CSV |
| `loso/benchmark_loso/` | `loso_benchmark.py` | CSP+LDA, CSP+SVM, FBCSP, EEGNet CSV + summary JSON |
| `loso/benchmark_loso_dl/` | `loso_dl_benchmark.py` | FBCSP, DeepConvNet, ShallowConvNet, EEGNet (含预处理) CSV + JSON |

`advanced_feature_benchmark` 和 `nn_models_benchmark` 均采用"内存聚合再落盘"的输出管路：

- 单被试 `*_umap3d.png`、`*_comparison_bar.png` 不再作为最终产物写入目录
- `subject_xx_metrics.json` 不再输出
- `--all-subjects` 主要产物为：
  - `all_subjects_metrics.csv`（9 被试 x 3 方法 = 27 行）
  - `all_subjects_umap3d_grid.png`（9 行 x 3 列）
  - `all_subjects_comparison_bar_grid.png`（3 x 3）
  - `all_subjects_summary.json`、`all_subjects_summary_bar.png`

---

## 快速开始

```bash
# 进入项目根目录
cd test_newPyEnv

# 列出所有已注册范式
.venv/bin/python BCICIV/BCICIV2a/pre-precess.py --list-paradigms

# 运行 TRCA/Wavelet/CNN 对比实验（单被试）
.venv/bin/python BCICIV/BCICIV2a/pre-precess.py --paradigm advanced_feature_benchmark --subject 1

# 运行全部 9 名被试
.venv/bin/python BCICIV/BCICIV2a/pre-precess.py --paradigm advanced_feature_benchmark --all-subjects

# 运行 FBCSP 混合降维可视化（启用 supervised UMAP）
.venv/bin/python BCICIV/BCICIV2a/pre-precess.py --paradigm hybrid_fbcsp_umap --subject 1 --supervised-umap

# 验证 DL 模型前向传播
.venv/bin/python BCICIV/BCICIV2a/test_nn_models_pipeline.py

# LOSO 跨被试泛化: CSP+LDA, CSP+SVM, FBCSP, EEGNet
.venv/bin/python BCICIV/BCICIV2a/paradigms/loso_benchmark.py

# LOSO DL 模型: FBCSP, DeepConvNet, ShallowConvNet, EEGNet (含统一预处理)
.venv/bin/python BCICIV/BCICIV2a/paradigms/loso_dl_benchmark.py

# Cross-session: E→T / T→T 评估
.venv/bin/python BCICIV/BCICIV2a/paradigms/cross_session_benchmark.py

# 直接运行各 benchmark (单被试)
.venv/bin/python BCICIV/BCICIV2a/paradigms/nn_models_benchmark.py --subject 1
.venv/bin/python BCICIV/BCICIV2a/paradigms/csp_lda_benchmark.py --subject 1
```

---

## 数据切分策略

| 切分方式 | 训练集 | 测试集 | 范式 | 说明 |
|---------|--------|--------|------|------|
| **within-subject** | 被试 X, Session T | 被试 X, Session E | within_subject 下全部范式 | BCICIV2a 竞赛标准方案 |
| **LOSO** | N-1 被试, Session T | 留出被试, Session **E** | `loso_benchmark.py` / `loso_dl_benchmark.py` | 跨被试泛化 (训练=T, 测试=E) |

within-subject 切分逻辑（`framework/data.py`）：
```python
is_train = metadata["session"].astype(str).str.contains("train")
is_test  = metadata["session"].astype(str).str.contains("test")
```

LOSO 切分逻辑：训练集聚合 N-1 被试的 Session T，测试集取留出被试的 Session E。统一预处理 (bandpass 0.5-40Hz + notch 50Hz + CAR) 在切分后应用。

---

## 新增范式

1. 在 `models/` 中实现算法（如有必要）
2. 在 `paradigms/` 下新建范式模块，实现 `build_config_from_namespace` 和 `run_from_config`
3. 在 `framework/registry.py` 的 `PARADIGM_REGISTRY` 中注册，指定 `default_result_group`
4. 结果自动写入 `results/{default_result_group}/`

---

## 当前已知问题

- **多 seed 实验缺失**：全范式仅使用单一随机种子（`random_state=42`），缺少多种子重复实验验证稳定性
- **DL 模型仍低于传统 CSP/LDA**：within-subject 最佳 DL (TCN 54.44%) 仍低于 CSP/LDA (56.91%)；LOSO 下 DL 优势明显 (42-43% vs FBCSP 33%)
- **LaBraM 微调仍未突破**：Adapter 微调 + 200 epoch 仍在 chance 附近 (28.94%)，110M 参数 vs 288 trial 矛盾未解决
- **Cross-session (E→T/T→T) 未运行**：`cross_session_benchmark.py` 已编写但未执行
- **Confusion matrix 已计算但未持久化落盘**
- **Negative control 不完整**：仅 TRCA/Wavelet/EEGNet 完成 label shuffle，其他模型待补充
