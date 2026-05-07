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
    ├── results/             实验输出
    │   ├── benchmark_trca_wavelet_cnn/
    │   ├── dim_reduction_hybrid_fbcsp/
    │   └── benchmark_nn_models/
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

**深度学习方法（III 期引入）：**

| 文件 | 方法 | 核心类 / 接口 |
|---|---|---|
| `tcn_model.py` | Temporal Convolutional Network | `TCNClassifier` · `train_tcn` · `predict_tcn` · `extract_tcn_features` |
| `atcnet_model.py` | Attention TCN (braindecode) | `ATCNetResult` · `train_atcnet` · `predict_atcnet` · `extract_atcnet_features` |
| `drsn_model.py` | Dilated Residual Shrinkage Network | `DRSNClassifier` · `train_drsn` · `predict_drsn` · `extract_drsn_features` |
| `labram_adapter.py` | LaBraM-Large (TorchEEG) | `LabramAdapter` · `setup_labram_pipeline` · `run_labram_experiment`（zero-shot 评估，fine-tuning 未实现） |

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

> **注意**：TCN / ATCNet / DRSN 已完成训练回路实现（含 LR scheduler、early stopping、checkpoint 落盘）；LaBraM 使用 TorchEEG 预训练权重进行 zero-shot 评估，fine-tuning 训练尚未实现。

### `results/` — 实验输出

| 子目录 | 对应范式 | 内容 |
|---|---|---|
| `benchmark_trca_wavelet_cnn/` | `advanced_feature_benchmark` | 全被试 CSV + UMAP 3D 总图 + comparison bar 总图 + 全被试汇总 |
| `dim_reduction_hybrid_fbcsp/` | `hybrid_fbcsp_umap` | UMAP 嵌入 .npz + 3D 可视化图 |
| `benchmark_nn_models/` | `nn_models_benchmark` | 全被试 CSV + aggregate summary JSON + summary bar + comparison bar grid + 单被试指标 JSON |

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
```

---

## 数据切分策略

当前所有范式使用 `framework/data.py` 中的 `load_subject_train_test()`，按 **Session T（训练）/ Session E（评估）** 切分。这是 BCICIV2a 竞赛标准的 within-subject cross-session 方案，**不是 LOSO**。

切分逻辑：
```python
is_train = metadata["session"].astype(str).str.contains("train")
is_test  = metadata["session"].astype(str).str.contains("test")
```

---

## 新增范式

1. 在 `models/` 中实现算法（如有必要）
2. 在 `paradigms/` 下新建范式模块，实现 `build_config_from_namespace` 和 `run_from_config`
3. 在 `framework/registry.py` 的 `PARADIGM_REGISTRY` 中注册，指定 `default_result_group`
4. 结果自动写入 `results/{default_result_group}/`

---

## 当前已知问题

- `LaBraM` 仅支持 zero-shot 评估，fine-tuning 训练回路尚未实现
- `advanced_benchmark.py` 中有 TCN / ATCNet / DRSN / LaBraM 的 stub 函数（`raise NotImplementedError`），实际实现在各自模型文件及 `nn_models_benchmark.py` 中
- DRSN 存在严重过拟合（train loss ~0.05 但 val loss 飙升至 3.0+），需加 dropout / weight decay 等正则化
- 全范式仅使用单一随机种子（`random_state=42`），缺少多种子重复实验
- 缺少 label shuffle negative control（`nn_models_benchmark` 已通过 `--shuffle-labels` 支持）
- confusion matrix 已在 benchmark 中计算但未持久化落盘
