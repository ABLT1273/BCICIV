"""
BCICIV2a 统一入口。

这个脚本把现有范式统一收口到一个入口：
- hybrid_fbcsp_umap
- advanced_feature_benchmark

运行示例：
test_newPyEnv/.venv/bin/python test_newPyEnv/BCICIV/BCICIV2a/pre-precess.py --paradigm hybrid_fbcsp_umap --subject 1
test_newPyEnv/.venv/bin/python test_newPyEnv/BCICIV/BCICIV2a/pre-precess.py --paradigm advanced_feature_benchmark --subject 1
test_newPyEnv/.venv/bin/python test_newPyEnv/BCICIV/BCICIV2a/pre-precess.py --paradigm advanced_feature_benchmark --all-subjects
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from framework.paths import get_result_group_dir
from framework.registry import PARADIGM_REGISTRY, load_paradigm_module
from framework.runtime import prepare_runtime_environment


def build_argument_parser() -> argparse.ArgumentParser:
    paradigm_choices = tuple(PARADIGM_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="BCICIV2a 统一实验入口。通过 --paradigm 选择需要运行的范式。"
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        choices=paradigm_choices,
        default="hybrid_fbcsp_umap",
        help="要运行的范式键。",
    )
    parser.add_argument("--subject", type=int, default=1, help="被试编号，默认 1。")
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="对支持该模式的范式，依次运行 1-9 号全部被试。",
    )
    parser.add_argument(
        "--supervised-umap",
        action="store_true",
        help="仅对 hybrid_fbcsp_umap 生效，启用 supervised UMAP。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="自定义结果输出目录；不传时写入范式默认结果目录。",
    )
    parser.add_argument("--show", action="store_true", help="保存后是否弹出图窗。")
    parser.add_argument(
        "--list-paradigms",
        action="store_true",
        help="只打印已注册范式及说明，不执行实验。",
    )
    return parser


def print_registered_paradigms() -> None:
    print("当前已注册范式：")
    for key, spec in PARADIGM_REGISTRY.items():
        print(f"- {key}: {spec.display_name}")
        print(f"  说明: {spec.description}")
        print(f"  输出目录组: {spec.default_result_group}")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    spec = PARADIGM_REGISTRY[args.paradigm]
    if args.output_dir is None:
        args.output_dir = get_result_group_dir(spec.default_result_group)
    return args


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.list_paradigms:
        print_registered_paradigms()
        return

    prepare_runtime_environment()

    import matplotlib

    if "--show" not in os.sys.argv:
        matplotlib.use("Agg")

    args = normalize_args(args)
    spec = PARADIGM_REGISTRY[args.paradigm]
    paradigm_module = load_paradigm_module(args.paradigm)

    print(f"当前范式: {spec.display_name}", flush=True)
    print(f"输出目录: {args.output_dir}", flush=True)

    config = paradigm_module.build_config_from_namespace(args)
    paradigm_module.run_from_config(config)


if __name__ == "__main__":
    main()
