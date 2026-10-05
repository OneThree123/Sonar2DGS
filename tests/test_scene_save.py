"""Exercise Scene.save without importing the CUDA scene module."""

import ast
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class SceneSaveTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        source = Path(__file__).resolve().parents[1] / "scene/__init__.py"
        tree = ast.parse(source.read_text())
        scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scene")
        save = next(node for node in scene.body if isinstance(node, ast.FunctionDef) and node.name == "save")
        self.sampled_export = Mock()
        self.mesh_export = Mock()
        namespace = dict(os=os, shutil=shutil, GAUSSIAN_MESH_ALPHA_MIN=.099,
                         export_sampled_pointcloud_from_gaussian_ply=self.sampled_export,
                         export_poisson_mesh_from_gaussian_ply=self.mesh_export)
        exec(compile(ast.Module(body=[save], type_ignores=[]), str(source), "exec"), namespace)
        self.save = namespace["save"]

        def save_ply(path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"GS parameters")

        self.gs_export = Mock(side_effect=save_ply)
        self.scene = SimpleNamespace(model_path=str(self.root),
                                     gaussians=SimpleNamespace(save_ply=self.gs_export))

    def test_gs_only_save_does_not_reconstruct_geometry(self):
        self.save(self.scene, 2, export_geometry=False)
        self.assertEqual((self.root / "point_cloud/iteration_2/point_cloud.ply").read_bytes(), b"GS parameters")
        self.gs_export.assert_called_once()
        self.sampled_export.assert_not_called()
        self.mesh_export.assert_not_called()

    def test_default_save_retains_geometry_exports(self):
        self.save(self.scene, 2)
        self.gs_export.assert_called_once()
        self.sampled_export.assert_called_once()
        self.mesh_export.assert_called_once()

    def test_existing_quality_geometry_is_reused(self):
        sampled = self.root / "samples.ply"
        mesh = self.root / "mesh.ply"
        sampled.write_bytes(b"sampled points")
        mesh.write_bytes(b"mesh")
        self.save(self.scene, 2, geometry_artifacts=dict(sampled_points_path=sampled, pred_mesh_path=mesh))
        output = self.root / "point_cloud/iteration_2"
        self.assertEqual((output / "sampled_surface_points.ply").read_bytes(), sampled.read_bytes())
        self.assertEqual((output / "poisson_mesh.ply").read_bytes(), mesh.read_bytes())
        self.gs_export.assert_called_once()
        self.sampled_export.assert_not_called()
        self.mesh_export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
