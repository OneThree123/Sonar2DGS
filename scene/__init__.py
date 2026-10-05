import os
import random
import json
import shutil
from pathlib import Path

import numpy as np

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos
from utils.gaussian_mesh import (
    GAUSSIAN_MESH_ALPHA_MIN,
    export_poisson_mesh_from_gaussian_ply,
    export_poisson_mesh_from_pointcloud_ply,
    export_sampled_pointcloud_from_gaussian_ply,
)


def camera_info_to_JSON(id, camera_info, split):
    c2w = np.asarray(camera_info.C2W, dtype=np.float64)

    return {
        "id": id,
        "split": split,
        "img_name": camera_info.image_name,
        "image_path": camera_info.image_path,
        "width": int(camera_info.img_width),
        "height": int(camera_info.img_height),
        "tx_width": int(camera_info.tx_width),
        "tx_height": int(camera_info.tx_height),
        "fovx": float(camera_info.FovX),
        "fovy": float(camera_info.FovY),
        "range_max": float(camera_info.range_max),
        "normalizing_method": getattr(camera_info, "normalizing_method", "all"),
        "data_type": getattr(camera_info, "data_type", ""),
        "position": c2w[:3, 3].tolist(),
        "rotation": c2w[:3, :3].tolist(),
    }


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = []
        self.test_cameras = []

        source_path = Path(args.source_path)
        if (source_path / "basic_info.json").exists():
            print("Found basic_info.json file, assuming sonar dataset.")
            scene_info = sceneLoadTypeCallbacks["Sonar"](
                args.source_path,
                getattr(args, "point_cloud_init", "auto"),
                getattr(args, "point_cloud_init_count", 100),
            )
        else:
            raise RuntimeError(
                f"Could not recognize scene type at {args.source_path}. "
                "Expected a sonar dataset with basic_info.json."
            )

        if not self.loaded_iter:
            input_ply_path = os.path.join(self.model_path, "input.ply")
            if scene_info.ply_path is not None:
                with open(scene_info.ply_path, "rb") as src_file, open(input_ply_path, "wb") as dest_file:
                    dest_file.write(src_file.read())
            else:
                storePly(input_ply_path, scene_info.point_cloud)

            json_cams = []
            camera_id = 0

            for cam in scene_info.train_cameras:
                json_cams.append(camera_info_to_JSON(camera_id, cam, "train"))
                camera_id += 1

            for cam in scene_info.test_cameras:
                json_cams.append(camera_info_to_JSON(camera_id, cam, "test"))
                camera_id += 1

            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)

        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "point_cloud.ply",
            ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            if (getattr(args, "save_reference_mesh", False)
                    and not getattr(args, "_quality_reference_managed", False)):
                self.save_reference_mesh(source_path / "dense_points3d.ply")
            print("Saving initial Gaussians")
            self.save(0)

    def save(self, iteration, geometry_artifacts=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        point_cloud_file = os.path.join(point_cloud_path, "point_cloud.ply")
        sampled_pointcloud_file = os.path.join(point_cloud_path, "sampled_surface_points.ply")
        poisson_mesh_file = os.path.join(point_cloud_path, "poisson_mesh.ply")

        self.gaussians.save_ply(point_cloud_file)
        if geometry_artifacts is not None:
            if geometry_artifacts.get("sampled_points_path"):
                shutil.copy2(geometry_artifacts["sampled_points_path"], sampled_pointcloud_file)
            if geometry_artifacts.get("pred_mesh_path"):
                shutil.copy2(geometry_artifacts["pred_mesh_path"], poisson_mesh_file)
            else:
                print(f"No valid evaluation mesh for iteration {iteration}; not repeating failed reconstruction.")
            return
        try:
            export_sampled_pointcloud_from_gaussian_ply(
                point_cloud_file,
                sampled_pointcloud_file,
                alpha_min=GAUSSIAN_MESH_ALPHA_MIN,
            )
        except Exception as exc:
            print(
                f"Warning: failed to save sampled surface point cloud for iteration {iteration}: {exc}. "
                "Training will continue."
            )

        try:
            export_poisson_mesh_from_gaussian_ply(
                point_cloud_file,
                poisson_mesh_file,
                alpha_min=GAUSSIAN_MESH_ALPHA_MIN,
            )
        except Exception as exc:
            print(
                f"Warning: failed to save Poisson mesh for iteration {iteration}: {exc}. "
                "Training will continue."
            )

    def save_reference_mesh(self, dense_point_cloud_path):
        dense_point_cloud_path = Path(dense_point_cloud_path)
        if not dense_point_cloud_path.exists():
            print(f"Reference dense point cloud not found, skipping mesh: {dense_point_cloud_path}")
            return

        reference_mesh_file = Path(self.model_path) / "point_cloud" / "reference_mesh.ply"
        print(f"Saving reference mesh from {dense_point_cloud_path}")
        try:
            export_poisson_mesh_from_pointcloud_ply(dense_point_cloud_path, reference_mesh_file)
        except Exception as exc:
            print(
                f"Warning: failed to save reference mesh from {dense_point_cloud_path}: {exc}. "
                "Training will continue."
            )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
