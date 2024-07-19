import joblib
import numpy as np
from smplx import SMPLLayer
from model.digital_man import DigitalMan

SMPLX_MODEL_PATH = f"/home/hume/PycharmProjects/Omnisense/ChestX_PCA/assets/smpl"
SMPLX_MODEL_GENDER = "neutral"


def load_digital_man(pkl_path: str) -> DigitalMan:
    pkl = joblib.load(pkl_path)
    smplx = SMPLLayer(model_path=SMPLX_MODEL_PATH, gender=SMPLX_MODEL_GENDER)
    faces = smplx.faces
    vertices = pkl["pred_vertices"][0]
    joints = pkl["pred_keypoints_3d"][0]
    camera_trans = pkl["pred_cam_t"][0]
    focal_length = pkl["focal_length"][0]
    image_size = np.array((640, 480))
    fpd_center = np.array((193.4, 324.5))
    pixel_spacing = 0.43 / 91.3
    return DigitalMan(faces, vertices, joints, camera_trans, focal_length, image_size, fpd_center, pixel_spacing)
