from argparse import ArgumentParser, Namespace
import os
import sys


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            arg_type = type(value)
            default_value = value if not fill_none else None

            if arg_type == bool:
                if shorthand:
                    group.add_argument("--" + key, "-" + key[0:1], default=default_value, action="store_true")
                else:
                    group.add_argument("--" + key, default=default_value, action="store_true")
            else:
                if shorthand:
                    group.add_argument("--" + key, "-" + key[0:1], default=default_value, type=arg_type)
                else:
                    group.add_argument("--" + key, default=default_value, type=arg_type)

    def extract(self, args):
        group = GroupParams()
        for key, value in vars(args).items():
            if key in vars(self) or "_" + key in vars(self):
                setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ["Intensity", "Alpha", "Normal", "Depth", "Edge", "Dist", "Curvature"]
        self.initial_opacity = 0.1#0.1
        self.point_cloud_init = "auto"
        self.point_cloud_init_count = 100
        self.save_reference_mesh = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        self.intensity_stats = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.stop_after_training_seconds = 0.0

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.reflection_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        # self.position_lr_init = 0.000128
        # self.position_lr_final = 0.0000016
        # self.position_lr_delay_mult = 0.01
        # self.position_lr_max_steps = 30_000

        # self.reflection_lr = 0.02
        # self.opacity_lr = 0.02
        # self.scaling_lr = 0.02
        # self.rotation_lr = 0.01

        self.percent_dense = 0.003
        self.lambda_dssim = 0.2
        self.lambda_empty = 0.0# was 1.0
        self.lambda_dist = 1.0 # was 1.0
        self.lambda_normal = 0.02  # was 0.02
        self.regularization_from_iter = 2_000
        self.loss_intensity_mode = "mean_match"#"normalized"#
        self.anti_noise = False
        self.anti_noise_min_component_size = 256
        self.anti_noise_min_neighbors = 6
        self.anti_noise_open_iterations = 1
        self.anti_noise_clear_top_ratio = 0.1
        self.opacity_cull = 0.01
        self.opacity_reset_value = 0.01
        self.opacity_reset_mode = "interval"

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.002
        self.densify_debug = False
        self.max_gaussians_initial = 100
        self.max_gaussians_anneal_target = 2_000
        self.max_gaussians_anneal_until_iter = 5_000
        self.max_gaussians = 50_000
        self.densify_direction_min_observations = 4
        self.densify_direction_min_consistency = 0.4
        self.clone_offset_min_factor = 0.2
        self.clone_offset_max_factor = 0.75
        self.lowpass_s_init = 10.0
        self.lowpass_s_final = 0.3
        self.lowpass_anneal_from_iter = 0
        self.lowpass_anneal_until_iter = 5_000

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdline_args = parser.parse_args(sys.argv[1:])
    cfgfile_string = "Namespace()"

    try:
        cfg_path = os.path.join(cmdline_args.model_path, "cfg_args")
        print("Looking for config file in", cfg_path)
        with open(cfg_path) as cfg_file:
            print("Config file found: {}".format(cfg_path))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
    except FileNotFoundError:
        pass

    cfg_args = eval(cfgfile_string)
    merged_args = vars(cfg_args).copy()

    for key, value in vars(cmdline_args).items():
        if value is not None:
            merged_args[key] = value

    return Namespace(**merged_args)
