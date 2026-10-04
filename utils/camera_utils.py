from scene.cameras import Camera
from utils.general_utils import PILtoTorch


def loadCam(args, id, cam_info):
    image = PILtoTorch(cam_info.image, (cam_info.img_width, cam_info.img_height))
    valid_mask = None
    if cam_info.mask is not None:
        valid_mask = PILtoTorch(cam_info.mask, (cam_info.img_width, cam_info.img_height))

    return Camera(
        colmap_id=cam_info.uid,
        C2W=cam_info.C2W,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        range_max=cam_info.range_max,
        image=image,
        tx_width=cam_info.tx_width,
        tx_height=cam_info.tx_height,
        img_height=cam_info.img_height,
        img_width=cam_info.img_width,
        valid_mask=valid_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        normalizing_method=getattr(cam_info, "normalizing_method", "all"),
        data_type=getattr(cam_info, "data_type", ""),
    )


def cameraList_from_camInfos(cam_infos, args):
    camera_list = []

    for id, cam_info in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, cam_info))

    return camera_list
