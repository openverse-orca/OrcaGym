import numpy as np
import h5py
import asyncio
# import websockets
# import av
import cv2
import io

def iter_dataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name, obj)


with h5py.File('episode_0.hdf5', 'r') as f:
    d = [
        '/observations/qpos',
        '/observations/qvel',
        '/observations/effort',
        '/action',
        '/base_action',
        '/observations/images/cam_high',
        '/observations/images/cam_left_wrist',
        '/observations/images/cam_right_wrist',
    ]
    f.visititems(iter_dataset)