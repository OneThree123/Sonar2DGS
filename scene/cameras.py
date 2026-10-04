import torch
from torch import nn

from utils.graphics_utils import getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        C2W,
        FoVx,
        FoVy,
        range_max,
        image,
        tx_width,
        tx_height,
        img_height,
        img_width,
        valid_mask,
        image_name,
        uid,
        data_device="cuda",
        normalizing_method="all",
        data_type="",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.image_name = image_name

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.range_max = range_max

        self.tx_width = int(tx_width)
        self.tx_height = int(tx_height)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.width = self.img_width
        self.height = self.img_height
        self.normalizing_method = str(normalizing_method).lower()
        self.data_type = str(data_type or "").upper()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)

        if valid_mask is not None:
            self.gt_alpha_mask = valid_mask.to(self.data_device)
        else:
            self.gt_alpha_mask = None

        c2w = torch.as_tensor(C2W, dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)

        self.C2W = c2w
        self.world_view_transform = w2c.transpose(0, 1).contiguous().cuda()

        near_range = max(1e-4, 1e-4 * float(self.range_max))
        self.projection_matrix = getProjectionMatrix(
            rmin=near_range,
            rmax=self.range_max,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).contiguous().cuda()

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = c2w[:3, 3].cuda()


class MiniCam:
    def __init__(
        self,
        fovy,
        fovx,
        range_max,
        tx_height,
        tx_width,
        img_height,
        img_width,
        world_view_transform,
        full_proj_transform,
        normalizing_method="all",
    ):
        self.FoVy = fovy
        self.FoVx = fovx
        self.range_max = range_max

        self.tx_height = int(tx_height)
        self.tx_width = int(tx_width)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.width = self.img_width
        self.height = self.img_height
        self.normalizing_method = str(normalizing_method).lower()

        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]
