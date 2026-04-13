# BCICIV2a Results

这个目录统一收纳 BCICIV2a 相关实验结果，按范式分子目录存放。

## 子目录

- `benchmark_trca_wavelet_cnn/`
  保存 TRCA、Wavelet、CNN 三种方法的单被试指标 JSON、对比柱状图及全被试汇总。

- `dim_reduction_hybrid_fbcsp/`
  保存 C3/C4 频域特征、FBCSP 特征的 UMAP 降维嵌入（.npz）与 3D 可视化图。

## 新增结果目录

在 `framework/registry.py` 注册新范式时，`default_result_group` 字段即为对应子目录名，
`framework/paths.py` 中的路径函数负责在首次写入时自动创建该目录。
