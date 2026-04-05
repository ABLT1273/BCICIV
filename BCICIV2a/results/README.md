# BCICIV2a Results

这个目录用于统一收纳 BCICIV2a 相关实验结果。

当前按范式分为两个子目录：

- `hybrid_reduction/`
  说明：保存 C3/C4 频域特征、FBCSP 特征以及 UMAP 可视化结果。
- `advanced_benchmark/`
  说明：保存 TRCA、Wavelet、CNN 等方法的单被试结果、全被试汇总与对比图。

统一框架入口：

- `pre-precess.py`
- `framework/runtime.py`
- `framework/data.py`
- `framework/paths.py`
- `framework/plotting.py`
- `framework/registry.py`
- `paradigms/hybrid_fbcsp_umap.py`
- `paradigms/advanced_benchmark.py`

当前已注册范式：

- `hybrid_fbcsp_umap`
  组件：`C3/C4 frequency` + `FBCSP` + `UMAP/supervised UMAP`
- `advanced_feature_benchmark`
  组件：`TRCA hybrid classifier` + `PyWavelets` + `EEGNet-style CNN`

后续新增范式时，建议：

1. 在 `framework/registry.py` 注册新的范式说明。
2. 复用 `framework/data.py` 的数据读取接口。
3. 在 `paradigms/` 下新增范式模块，并提供 `build_config_from_namespace` / `run_from_config`。
4. 将结果写入 `results/` 下对应的新子目录。
