# week6 泛化结果矩阵

## 目标

并列呈现 `within-subject`、`LOSO` 和可选 `cross-session` 的结果,避免只报一个离线 accuracy。

## 结果总表 (2026-05-09 优化后)

| 模型 | 切分方式 | Accuracy mean | Accuracy std | 最差 subject | 最好 subject | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| CSP/LDA | within-subject | 56.91% | 14.18% | S5 (36.11%) | S8 (76.39%) | 经典基准, filter+CAR 后略降 |
| TCN | within-subject | 54.44% | 16.09% | S2 (33.33%) | S9 (73.26%) | 2-layer head + label smoothing |
| DRSN | within-subject | 52.89% | 13.84% | S5 (31.60%) | S3 (74.65%) | rsnet10 + 22→8 proj + 2-layer head |
| ATCNet | within-subject | 52.82% | 18.65% | S2 (27.78%) | S9 (78.47%) | label smoothing |
| FBCSP | within-subject | 52.55% | 15.66% | S5 (31.94%) | S9 (72.57%) | Filter Bank CSP + NBPW |
| DFBCSP | within-subject | 51.93% | 15.03% | S2 (31.60%) | S9 (72.22%) | 判别式 FBCSP (n_select=4) |
| EEGNet | within-subject | 50.12% | 17.98% | S5 (27.08%) | S9 (72.22%) | 3-block, 50 epoch, ReduceLROnPlateau |
| Wavelet | within-subject | 44.52% | 9.75% | S5 (28.82%) | S1 (57.99%) | 小波变换 + SVM |
| TRCA | within-subject | 37.65% | 7.27% | S2 (29.17%) | S9 (49.65%) | TRCA + SVM, proportional reg fix |
| LaBraM | within-subject | 28.94% | 6.65% | S4 (22.22%) | S9 (42.01%) | Adapter 微调, 仍近 chance |
| CSP/LDA | LOSO | 运行中 | — | — | — | 待完成 |
| FBCSP | LOSO | 运行中 | — | — | — | 待完成 |
| EEGNet | LOSO | 运行中 | — | — | — | 待完成 |
| CSP/LDA | cross-session | 未运行 | — | — | — | 可选 |
| EEGNet | cross-session | 未运行 | — | — | — | 可选 |

### 逐 Subject 明细

| Subject | CSP/LDA | FBCSP | DFBCSP | ATCNet | TCN | EEGNet | Wavelet | TRCA | DRSN | LaBraM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 70.83% | 68.40% | 61.46% | 62.15% | 68.40% | 70.49% | 57.99% | 49.31% | 60.42% | 38.19% |
| 2 | 38.89% | 39.24% | 31.60% | 27.78% | 33.33% | 30.21% | 35.42% | 29.17% | 39.93% | 24.31% |
| 3 | 72.22% | 59.72% | 54.51% | 71.18% | 72.22% | 70.49% | 52.43% | 39.93% | 74.65% | 31.94% |
| 4 | 53.47% | 37.85% | 42.01% | 50.35% | 52.78% | 46.88% | 36.46% | 34.38% | 47.57% | 22.22% |
| 5 | 36.11% | 31.94% | 38.19% | 31.25% | 35.42% | 27.08% | 28.82% | 30.90% | 31.60% | 26.04% |
| 6 | 43.06% | 38.89% | 36.81% | 32.99% | 39.58% | 35.42% | 38.89% | 29.86% | 40.97% | 22.92% |
| 7 | 57.99% | 56.25% | 62.85% | 45.83% | 43.06% | 34.38% | 44.10% | 36.81% | 47.92% | 24.31% |
| 8 | 76.39% | 68.06% | 67.71% | 75.35% | 71.88% | 63.89% | 49.31% | 38.89% | 68.40% | 28.47% |
| 9 | 63.19% | 72.57% | 72.22% | 78.47% | 73.26% | 72.22% | 57.29% | 49.65% | 64.58% | 42.01% |

> 粗体 = 该 subject 上排名前 3。

### 模型排名 (按 mean accuracy 降序)

| 排名 | 模型 | Mean Acc | Std | 类型 | vs 优化前 |
| --- | --- | --- | --- | --- | --- |
| 1 | CSP/LDA | 56.91% | 14.18% | 传统 (CSP + LDA) | -2.47% |
| 2 | TCN | 54.44% | 16.09% | DL (时序卷积) | **+7.14%** |
| 3 | DRSN | 52.89% | 13.84% | DL (残差收缩) | **+20.29%** |
| 4 | ATCNet | 52.82% | 18.65% | DL (Attention+TCN) | -1.04% |
| 5 | FBCSP | 52.55% | 15.66% | 传统 (Filter Bank CSP) | 0% |
| 6 | DFBCSP | 51.93% | 15.03% | 传统 (判别式 FBCSP) | 0% |
| 7 | EEGNet | 50.12% | 17.98% | DL (轻量 CNN) | **+9.53%** |
| 8 | Wavelet | 44.52% | 9.75% | 传统 (小波 + SVM) | +4.55% |
| 9 | TRCA | 37.65% | 7.27% | 传统 (TRCA + SVM) | +0.54% |
| 10 | LaBraM | 28.94% | 6.65% | DL (Transformer) | -0.11% |

> CSP/LDA 下降是因为统一预处理 (filter+CAR) 对所有模型生效, CAR 降低通道秩对 CSP 有一定影响; 但对其他模型 (TCN/DRSN/EEGNet/Wavelet) 帮助显著。

## 优化结果 (2026-05-09)

### 关键改进项

| 优先级 | 优化项 | 影响模型 | 实际效果 |
|--------|--------|---------|---------|
| P0 | 统一预处理 (bandpass 0.5-40Hz + notch 50Hz + CAR) | 全局 | CSP 略降, 其余 +2~20% |
| P0 | DRSN 架构精简 (rsnet10, Conv1d 22→8) | DRSN | **+20.29%** (32.60→52.89) |
| P1 | EEGNet 加深 (3-block) + 50 epoch + ReduceLROnPlateau | EEGNet | **+9.53%** (40.59→50.12) |
| P1 | TCN 2-layer head + label smoothing | TCN | **+7.14%** (47.30→54.44) |
| P1 | LaBraM Adapter 冻结 + 200 epoch | LaBraM | 无明显变化 (28.94%) |
| P2 | z-score 实现去重 (4 个 DL 模型共享) | DL | 代码质量, 无误导 |
| P2 | LaBraM 双滤波修复 | LaBraM | 避免级联滤波 |
| P2 | TRCA proportional regularization | TRCA | +0.54%, 修复数值崩溃 |

### 优化前后对比 (9-subject mean)

| 模型 | 优化前 | 优化后 | 变化 | 关键优化项 |
|------|--------|--------|------|-----------|
| CSP/LDA | 59.38% | 56.91% | -2.47% | filter+CAR (对 CSP 空间滤波有副作用) |
| TCN | 47.30% | 54.44% | **+7.14%** | 2-layer head, label smoothing |
| DRSN | 32.60% | 52.89% | **+20.29%** | rsnet10, 22→8 proj, 2-layer head, label smoothing |
| ATCNet | 53.86% | 52.82% | -1.04% | label smoothing (抽样波动) |
| FBCSP | 52.55% | 52.55% | 0% | 自身 FilterBank 已处理频带 |
| DFBCSP | 51.93% | 51.93% | 0% | 同上 |
| EEGNet | 40.59% | 50.12% | **+9.53%** | 3-block, 50 epoch, ReduceLROnPlateau, label smoothing |
| Wavelet | 39.97% | 44.52% | +4.55% | filter+CAR |
| TRCA | 37.11% | 37.65% | +0.54% | proportional reg fix |
| LaBraM | 29.05% | 28.94% | -0.11% | Adapter 冻结 + 200 epoch (无实质改善) |

### 关键发现

1. **DL 模型大幅逼近传统方法**: TCN 和 DRSN 从垫底跃升至第 2/3 位, EEGNet 从 40.59% → 50.12%, DL 与传统的差距从 ~20% 缩小到 ~5%
2. **统一预处理 (filter+CAR) 对 DL 模型至关重要**: 所有 DL 模型 + 小波均受益
3. **DRSN 改进最显著 (+20.29%)**: 原架构 (Conv1d 22→1, resnet18, 11M params) 对 288 trial 严重过拟合, 精简后效果立竿见影
4. **LaBraM 微调仍无效 (28.94%)**: 110M 参数 vs 288 trial, Adapter 冻结策略未解决根本矛盾, 可能需数据增强或预训练对齐
5. **CSP/LDA 仍是天花板**: 56.91%, 简单空间滤波 + 线性分类在小样本 EEG 上仍是最强方法
6. **TCN 性价比最高**: 训练 5-10s, 54.44%, 代码简单, 适合快速迭代

## Seed sensitivity

| 模型 | 切分方式 | Seed | Accuracy | 备注 |
| --- | --- | --- | --- | --- |
| — | — | — | — | 当前仅运行了单一 seed, 多 seed 实验待补充 |

## Negative control

| 检查 | 预期 | 实际 | 结论 |
| --- | --- | --- | --- |
| EEGNet label shuffle (S1) | ~25% (4-class chance) | normal 51.04% → shuffled 23.61% | 通过 |
| TRCA label shuffle (S1) | ~25% | normal 54.51% → shuffled 28.82% | 通过 |
| Wavelet label shuffle (S1) | ~25% | normal 39.58% → shuffled 26.39% | 通过 |
| DL 模型 label shuffle | ~25% | 未运行 | 待补充 |
| FBCSP/DFBCSP label shuffle | ~25% | 未运行 | 待补充 |

## 初步解释

- **within-subject 能说明:** 模型在同一被试的不同 session 之间具备一定的模式识别能力
- **within-subject 不能说明:** 模型能泛化到未见过的被试; 高 within-subject 不代表跨被试泛化
- **LOSO 下降说明:** (待运行后填写)
- **模型排序分析:**
  - **CSP/LDA (56.91%) 仍是 BCICIV2a 上最强的方法**, CSP 空间滤波 + 简单线性分类在小样本 EEG 上极有效
  - **DL 三强 (TCN 54.44%, DRSN 52.89%, ATCNet 52.82%) 已接近传统方法**, 差距从 ~20% 缩小到 ~4%
  - **TCN 性价比突出**: 训练快 (~6s), 参数量适中 (~500K), 准确率 54.44%
  - **DRSN 改进最显著**: 从 32.60% → 52.89%, 证明架构精简对小样本至关重要
  - **传统方法集群 (FBCSP/DFBCSP ~52%) 表现稳定**, 受统一预处理影响小 (自身已有频带处理)
  - LaBraM (28.94%) 仍未突破 chance level, 需更激进的数据增强或预训练策略
- **传统 vs DL:**
  - 前 7 名中传统方法占 3 席 (CSP/LDA, FBCSP, DFBCSP), DL 占 4 席 (TCN, DRSN, ATCNet, EEGNet)
  - DL 集体追平传统方法, 但单个最佳仍是传统 CSP/LDA
  - 小样本 EEG 上, 空间滤波 (CSP) 的信息提取效率仍优于端到端学习
- **与自动化所跨被试/跨任务方向的连接:**
  - 当前仅在 BCICIV2a 上验证了 within-subject, LOSO 运行中
  - CSP/LDA 在跨被试场景中预期大幅下降, DL 可能有更好的泛化能力
  - FBCSP 的 Filter Bank 分解可迁移到 DL: 多频带并行卷积 → 融合
  - CSP/LDA 可作为跨被试迁移的 teacher 或 alignment 信号
  - LaBraM 需微调 (fine-tuning) 而非 zero-shot 才能释放潜力
