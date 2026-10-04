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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    reflections: np.ndarray
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(rmin, rmax, fovX, fovY):
    """
    Sonar projection matrix.

    Sonar local coordinates:
        +X: imaging center / forward range direction
        +Y: yaw direction, controlled by fovX
        +Z: pitch direction, controlled by fovY

    This function returns the column-vector mathematical matrix.
    Camera code should keep transposing it before passing to CUDA, because
    the rasterizer uses row-vector style multiplication.
    """
    tanHalfFovX = math.tan(fovX * 0.5)
    tanHalfFovY = math.tan(fovY * 0.5)

    P = torch.zeros(4, 4)
    range_sign = 1.0

    # clip.x / clip.w = y / (x * tan(fovX / 2))
    # clip.y / clip.w = z / (x * tan(fovY / 2))
    # clip.w = x
    P[0, 1] = -1.0 / tanHalfFovX
    P[1, 2] = -1.0 / tanHalfFovY

    P[2, 0] = range_sign * rmax / (rmax - rmin)
    P[2, 3] = -(rmax * rmin) / (rmax - rmin)
    P[3, 0] = range_sign

    return P


# def fov2focal(fov, pixels):
#     return pixels / (2 * math.tan(fov / 2))

# def focal2fov(focal, pixels):
#     return 2*math.atan(pixels/(2*focal))
