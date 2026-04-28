#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0  # 0.0 previously. Changed to 100.0 to match the paper.
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser):
    """
    读取命令行参数，同时（如果存在）尝试从 <model_path>/cfg_args 里
    加载额外的配置并 merge 进去。

    为了兼容我们在 reinforce 训练时写入的 cfg_args（可能是
    "<arguments.GroupParams object at ...>" 这种不能 eval 的字符串），
    如果 eval 失败，就直接忽略 cfg_args，仅使用命令行参数。
    """
    # 先正常解析命令行
    args = parser.parse_args()

    # 没有 model_path 就没法找 cfg_args，直接返回
    model_path = getattr(args, "model_path", None)
    if model_path is None:
        return args

    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return args

    # 读 cfg_args 文件
    try:
        with open(cfg_path, "r") as f:
            cfgfile_string = f.read()
    except OSError:
        # 读不到就直接用命令行参数
        return args

    # 🔴 关键：这里 eval 有可能炸（比如内容是 "<arguments.GroupParams ...>"）
    try:
        cfg_obj = eval(cfgfile_string)
    except Exception:
        # eval 失败，说明这个 cfg_args 不是 Python literal，直接忽略即可
        return args

    # 如果是 dict，就把里面的键值合并到 args 里
    if isinstance(cfg_obj, dict):
        for k, v in cfg_obj.items():
            # 避免把完全无关的东西硬塞进去
            if not hasattr(args, k):
                setattr(args, k, v)

    # 如果是其他类型（Namespace / 自定义对象），这里就先不处理，至少不报错
    return args

