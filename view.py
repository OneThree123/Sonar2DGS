import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, network_gui
from utils.image_utils import render_net_image
import torch

def view(dataset, pipe, iteration):
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    if len(scene.getTrainCameras()) > 0:
        network_gui.set_sonar_defaults(scene.getTrainCameras()[0])
    elif len(scene.getTestCameras()) > 0:
        network_gui.set_sonar_defaults(scene.getTestCameras()[0])

    while True:
        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)

            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifier, render_mode = network_gui.receive()

                    if custom_cam is not None:
                        render_pkg = render(custom_cam, gaussians, pipe, scaling_modifier)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, 0.0, 1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )

                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)

                except Exception:
                    network_gui.conn = None


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--iteration', type=int, default=30000)
    args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    network_gui.init(args.ip, args.port)
    
    view(lp.extract(args), pp.extract(args), args.iteration)

    print("\nViewing complete.")