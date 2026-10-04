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
import traceback
import socket
import json
import struct
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)




def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def send_json_data(conn, data):
    # Serialize the list of strings to JSON
    serialized_data = json.dumps(data)
    # Convert the serialized data to bytes
    bytes_data = serialized_data.encode('utf-8')
    # Send the length of the serialized data first
    conn.sendall(struct.pack('I', len(bytes_data)))
    # Send the actual serialized data
    conn.sendall(bytes_data)

def try_connect(render_items):
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        # print(f"\nConnected by {addr}")
        conn.settimeout(None)
        send_json_data(conn, render_items)
    except Exception as inst:
        pass
        # raise inst
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify, metrics):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))
    send_json_data(conn, metrics)

sonar_defaults = {}

def set_sonar_defaults(camera):
    sonar_defaults.clear()
    sonar_defaults.update({
        "fovx": float(camera.FoVx),
        "fovy": float(camera.FoVy),
        "range_max": float(camera.range_max),
        "tx_width": int(camera.tx_width),
        "tx_height": int(camera.tx_height),
        "normalizing_method": getattr(camera, "normalizing_method", "all"),
    })

def _message_value(message, keys, default=None, required=True):
    for key in keys:
        if key in message and message[key] is not None:
            return message[key]
    if default is not None:
        return default
    if required:
        raise KeyError(f"Viewer message is missing one of: {keys}")
    return None


def receive():
    message = read()

    width = int(_message_value(message, ("resolution_x",)))
    height = int(_message_value(message, ("resolution_y",)))

    do_training = bool(message.get("train", True))
    keep_alive = bool(message.get("keep_alive", True))
    scaling_modifier = float(message.get("scaling_modifier", 1.0))
    render_mode = int(message.get("render_mode", 0))

    custom_cam = None
    if width > 0 and height > 0:
        fovx = float(_message_value(message, ("fov_x", "fovx"), sonar_defaults.get("fovx")))
        fovy = float(_message_value(message, ("fov_y", "fovy"), sonar_defaults.get("fovy")))
        range_max = float(_message_value(message, ("range_max", "z_far"), sonar_defaults.get("range_max")))

        tx_width = int(message.get("tx_width", sonar_defaults.get("tx_width", width)))
        tx_height = int(message.get("tx_height", sonar_defaults.get("tx_height", height)))
        normalizing_method = message.get(
            "normalizing_method",
            sonar_defaults.get("normalizing_method", "all"),
        )
        if normalizing_method is None:
            normalizing_method = "all"
        normalizing_method = str(normalizing_method).lower()

        world_view_transform = torch.reshape(
            torch.tensor(message["view_matrix"], dtype=torch.float32), (4, 4)
        ).cuda()

        world_view_transform[:, 1] = -world_view_transform[:, 1]
        world_view_transform[:, 2] = -world_view_transform[:, 2]

        near_range = max(1e-4, 1e-4 * range_max)
        projection_matrix = getProjectionMatrix(
            rmin=near_range,
            rmax=range_max,
            fovX=fovx,
            fovY=fovy,
        ).transpose(0, 1).contiguous().cuda()

        full_proj_transform = (
            world_view_transform.unsqueeze(0)
            .bmm(projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

        custom_cam = MiniCam(
            fovy=fovy,
            fovx=fovx,
            range_max=range_max,
            tx_height=tx_height,
            tx_width=tx_width,
            img_height=height,
            img_width=width,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            normalizing_method=normalizing_method,
        )

    return custom_cam, do_training, keep_alive, scaling_modifier, render_mode
