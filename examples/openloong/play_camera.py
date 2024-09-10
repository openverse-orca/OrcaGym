import numpy as np
import h5py
import asyncio
# import websockets
# import av
import cv2
import io


# with h5py.File('episode_0.hdf5', 'r') as f:
with h5py.File('openloong_xbox_control_record.h5', 'r') as f:    
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    for name in camera_names:
        for frame in f['/observations/images/' + name]:
            cv2.imshow(name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()