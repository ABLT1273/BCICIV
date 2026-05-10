# BCICIV2a 模型性能优化方案

## Context

基于 10 个模型在 BCICIV2a (4-class MI) 上的 within-subject 结果:
- CSP/LDA: 59.38% | ATCNet: 53.86% | FBCSP: 52.55% | DFBCSP: 51.93% | TCN: 47.30%
- EEGNet: 40.59% | Wavelet: 39.97% | TRCA: 37.11% | DRSN: 32.60% | LaBraM: 29.05%

核心问题: 训练数据极少 (288 trials/subject), DL 模型普遍欠拟合/过拟合, 传统方法缺少预处理。

---

## 一、数据预处理优化 (优先级: 最高, 收益最大)

### 1.1 通用预处理 pipeline (所有模型受益)

| 步骤 | 当前状态 | 建议 | 预期收益 |
|------|---------|------|---------|
| 带通滤波 0.5-40Hz | 仅 FBCSP/LaBraM 有 | 所有模型统一应用 | +2-5% Acc |
| 陷波滤波 50Hz | 仅 LaBraM 有 | 所有模型统一应用 | 去噪 |
| CAR (共平均参考) | 无 | 所有模型统一应用 | +1-3% Acc |
| 时间窗扩展 0.0-4.0s | 0.5-2.5s | 包含准备期 ERD | +2-5% Acc |
| Z-score 标准化 | DL 模型有, 传统模型无 | 传统模型也加入 | +1-2% Acc |

**实现**: 在 `framework/data.py` 的 `load_subject_train_test` 中增加预处理参数 (filter, CAR, normalize), 各 benchmark 脚本按需启用。

### 1.2 数据增强 (DL 模型关键)

当前 288 trial 训练 10K~110M 参数的模型, 严重数据不足。

| 增强方法 | 说明 | 预期收益 |
|---------|------|---------|
| 滑动窗口裁剪 | 从 4s epoch 中随机裁 2s 子窗口, 放大 5-10x | +5-10% Acc |
| 高斯噪声注入 | SNR=10-20dB, 每 trial 生成 2-3 变体 | +3-5% Acc |
| 通道 Dropout | 随机丢弃 10-20% 通道 | +2-3% Acc (正则化) |
| Mixup | 样本间线性插值 (alpha=0.2) | +2-4% Acc |

**实现**: 新建 `framework/augmentation.py`, 在 PyTorch DataLoader 中以 transform 形式应用。

### 1.3 伪迹处理 (可选)

- ICA 去除眼电/肌电分量 (需人工审核, 暂不自动化)
- 基于幅值的 epoch 拒绝 (±100μV 阈值)

---

## 二、模型架构优化

### 2.1 DRSN 结构改进 (当前 32.60% → 目标 40-45%)

**问题**: (1) Conv1d(22→1) 压缩比过大 (2) 11M 参数严重过拟合 288 trial

**方案**:
```python
# 替换: Conv1d(22, 1, kernel=1)     # 22→1 太激进
# 改为: Conv1d(22, 8, kernel=1)     # 22→8, 保留空间信息
#        + 减少 resnet 层: [2,2,2,2] → [1,1,1,1] (参数量减半)
```
- 替换 backbone: `rsnet18` → 轻量 `rsnet10` (约 2-3M 参数)
- 增加 Dropout(0.3) 在 FC 层

### 2.2 LaBraM 微调策略改进 (当前 29.05% → 目标 40-50%)

**问题**: 110M 参数模型, 全量微调 + 288 trial = 严重过拟合

**方案**:
- **Adapter tuning**: 冻结预训练权重, 仅在每层插入 Adapter (bottleneck dim=32) → 可训练参数降至 ~2M
- **增加训练 epoch**: 50 → 200 (Adapter 收敛更慢但更稳定)
- **数据增强配合**: 滑动窗口 + 噪声注入放大训练集 10x
- **学习率**: Adapter 用 5e-4, head 用 1e-3

### 2.3 EEGNet 加深 (当前 40.59% → 目标 45-50%)

**问题**: 仅 12 epoch + 2 blocks, 表达能力不足

**方案**:
- 增加 Block 3: 与 Block 2 相同结构 (可分离卷积)
- Epochs: 12 → 50
- 配合 ReduceLROnPlateau (当前用 CosineAnnealing 对短训练不利)

### 2.4 CSP 预处理 + DL (混合方案)

**灵感**: FBCSP 的 Filter Bank 思路 → DL

**方案**: 对 DL 模型输入前加 CSP 空间滤波
```python
# CSP(n_components=8) 应用于原始数据 → 输出 (n_trials, 32, n_times)
# (4 classes × 2 pairs/class × 4 components/pair = 32 CSP features)
# 将 CSP 特征作为 DL 模型输入, 替代原始 22 通道
```
- 预期收益: +5-10% Acc (空间滤波是 MI 的核心)

---

## 三、超参数调优

### 3.1 各模型关键超参数搜索

| 模型 | 调优参数 | 搜索范围 | 当前值 |
|------|---------|---------|--------|
| CSP/LDA | n_components | [2,4,6,8] | 4 |
| FBCSP/DFBCSP | m (CSP pairs), k (features), n_select (bands) | m=[2,3,4], k=[2,4,6,8], n_select=[2,4,6] | m=2,k=4,n_select=4 |
| EEGNet | temporal_filters, dropout, lr | F1=[8,16,32], drop=[0.25,0.5], lr=[1e-3,5e-4,1e-4] | F1=16,drop=0.25,lr=1e-3 |
| ATCNet | n_windows, att_dropout, n_heads | n_win=[3,5,7], att_drop=[0.3,0.5], n_heads=[2,4] | 5, 0.5, 2 |
| TCN | num_channels, kernel_size, dropout | ch=[[32,64],[32,64,128]], k=[3,5,7], drop=[0.1,0.25,0.5] | [32,64,128], 5, 0.25 |
| DRSN | channel_proj, num_blocks, lr | proj=[4,8,16], blocks=[resnet10,resnet18], lr=[1e-4,5e-4] | 1, resnet18, 1e-3 |
| LaBraM | adapter_dim, lr, epochs | dim=[16,32,64], lr=[1e-5,5e-5,1e-4], epoch=[50,100,200] | 无adapter, 1e-4, 50 |

### 3.2 调优策略

- 每个模型选择 S1 (最好) 和 S2 (最差) 两个被试做 grid search
- 最佳超参数应用到所有 9 个被试
- 使用 5-fold cross-validation on Session T 而非固定 80/20 split

---

## 四、分类器优化

### 4.1 传统模型分类器替换

| 模型 | 当前 | 建议 | 预期收益 |
|------|------|------|---------|
| CSP | OneVsRest(LDA) | OneVsRest(RBF-SVM, C=1.0) | +2-4% |
| FBCSP/DFBCSP | NBPW (Parzen Window) | 集成: NBPW + LightGBM + SVM 投票 | +2-5% |
| Wavelet/TRCA | RBF-SVM | 保持不变 (已经最优) | — |

### 4.2 DL 模型分类头优化

- 当前: 单层 `Linear(embedding_dim, 4)` 
- 改进: `Linear → BatchNorm → ReLU → Dropout → Linear` (2-layer head)
- 标签平滑 (label smoothing=0.1): 减少过拟合
- 温度缩放校准: 后处理步骤, 不改变 Acc 但提高概率质量

### 4.3 OVR 策略对比

FBCSP/DFBCSP 使用 OVR (4 个二分类器), TRCA 也使用 OVR。可以尝试:
- **OVO (One-vs-One)**: 6 个二分类器, 对左右手混淆可能有帮助
- 对比 OVR vs OVO 在 S1 上的表现

---

## 五、实施优先级

| 优先级 | 优化项 | 难度 | 预期总收益 | 涉及文件 |
|--------|--------|------|-----------|---------|
| P0 | 通用预处理 (filter+CAR+时间窗) | 低 | +5-10% 全局 | `framework/data.py` |
| P0 | CSP预处理 + DL | 中 | +5-10% DL | 各 model adapter |
| P0 | DRSN 结构精简 | 中 | +10-15% DRSN | `models/drsn_model.py`, `official_drsn/` |
| P1 | 数据增强 pipeline | 中 | +5-10% DL | 新建 `framework/augmentation.py` |
| P1 | LaBraM Adapter 微调 | 中 | +10-20% LaBraM | `models/labram_adapter.py` |
| P1 | 超参数 grid search | 高 | +3-8% 全局 | 新建 `paradigms/hparam_search.py` |
| P2 | EEGNet 加深 + 更多 epoch | 低 | +3-5% EEGNet | `models/deep_cnn_features.py` |
| P2 | 分类器替换 (SVM/LightGBM) | 低 | +2-5% 传统模型 | 各 benchmark 脚本 |
| P2 | 标签平滑 + 2-layer head | 低 | +1-3% DL | 各 model adapter |

---

## 六、验证方法

1. **A/B 测试**: 每个优化项在 S1 (easy) + S2 (hard) 上对比
2. **完整运行**: 确认有效的优化 → 所有 9 个被试全量运行
3. **Negative control**: 对修改后的模型做 label shuffle (预期 ~25%)
4. **更新泛化矩阵**: `week6_泛化结果矩阵.md` 新增优化后结果列
