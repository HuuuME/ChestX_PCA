from typing import Optional, List

import numpy as np


def xyz2uv(points: np.array, camera_translation, focal_length, camera_center) -> np.array:
    K = np.zeros([3, 3], dtype=np.float32)
    K[0, 0] = focal_length[0]
    K[1, 1] = focal_length[1]
    K[:-1, 2] = camera_center
    K[2, 2] = 1.

    points_copy = points.copy()
    points_copy += camera_translation
    points_copy /= np.expand_dims(points_copy[:, -1], axis=-1)

    pixels = np.array([np.dot(point, K.transpose()) for point in points_copy])
    return pixels[:, :-1]


def perpendicular(line: np.array, point: np.array) -> np.array:
    x1, y1, x2, y2 = line[0][0][0], line[0][0][1], line[1][0][0], line[1][0][1]
    x3, y3 = point[0][0], point[0][1]
    k1 = (y2 - y1) / (x2 - x1 + 1e-9)
    b1 = y1 - k1 * x1
    k2 = -1/k1
    b2 = y3 - k2 * x3
    x = - (b1 - b2) / (k1 - k2 + 1e-9)
    y = (b2 * k1 - b1 * k2) / (k1 - k2 + 1e-9)
    return np.array([x, y])


class Limb:

    def __init__(self, joints: np.array, links: np.array):
        self.joints = joints
        self.links = links


class BasePoints:

    def __init__(self, x: Optional[List] = None, y: Optional[List] = None, z: Optional[List] = None):
        self.x = x
        self.y = y
        self.z = z


class BodyPart:

    def __init__(self, vertices_index: List, faces_index: List, all_faces: np.array):

        self.vertices_index = vertices_index
        self.faces_index = faces_index
        self.all_faces = all_faces
        self.base_points: Optional[BasePoints] = None
        self.faces = []
        for index in self.faces_index:
            face = self.all_faces[index]
            self.faces.append(np.array([
                self.vertices_index.index(face[0]),
                self.vertices_index.index(face[1]),
                self.vertices_index.index(face[2])
            ]))

    def set_base_points(self, base_points: BasePoints):
        self.base_points = base_points


class DigitalMan:

    JOINTS_MAPPER = {
        "Pelvis": 0,
        "LHip": 1,
        "RHip": 2,
        "Spine1": 3,
        "LKnee": 4,
        "RKnee": 5,
        "Spine2": 6,
        "LAnkle": 7,
        "RAnkle": 8,
        "Spine3": 9,
        "LeftFoot": 10,
        "RightFoot": 11,
        "Neck": 12,
        "LeftCollar": 13,
        "RightCollar": 14,
        "Jaw": 15,
        "LShoulder": 16,
        "RShoulder": 17,
        "LElbow": 18,
        "RElbow": 19,
        "LWrist": 20,
        "RWrist": 21,
        "LHand": 22,
        "RHand": 23,
    }

    NECK_JOINT_INDEX = JOINTS_MAPPER["Neck"]

    RIGHT_SHOULDER_JOINT_INDEX = JOINTS_MAPPER["RShoulder"]
    RIGHT_ELBOW_JOINT_INDEX = JOINTS_MAPPER["RElbow"]
    RIGHT_WRIST_JOINT_INDEX = JOINTS_MAPPER["RWrist"]
    RIGHT_HAND_JOINT_INDEX = JOINTS_MAPPER["RHand"]

    LEFT_SHOULDER_JOINT_INDEX = JOINTS_MAPPER["LShoulder"]
    LEFT_ELBOW_JOINT_INDEX = JOINTS_MAPPER["LElbow"]
    LEFT_WRIST_JOINT_INDEX = JOINTS_MAPPER["LWrist"]
    LEFT_HAND_JOINT_INDEX = JOINTS_MAPPER["LHand"]

    PELVIS_JOINT_INDEX = JOINTS_MAPPER["Pelvis"]

    RIGHT_HIP_JOINT_INDEX = JOINTS_MAPPER["RHip"]
    RIGHT_KNEE_JOINT_INDEX = JOINTS_MAPPER["RKnee"]
    RIGHT_ANKLE_JOINT_INDEX = JOINTS_MAPPER["RAnkle"]
    RIGHT_FOOT_JOINT_INDEX = JOINTS_MAPPER["RightFoot"]

    LEFT_HIP_JOINT_INDEX = JOINTS_MAPPER["LHip"]
    LEFT_KNEE_JOINT_INDEX = JOINTS_MAPPER["LKnee"]
    LEFT_ANKLE_JOINT_INDEX = JOINTS_MAPPER["LAnkle"]
    LEFT_FOOT_JOINT_INDEX = JOINTS_MAPPER["LeftFoot"]

    SPINE1_JOINT_INDEX = JOINTS_MAPPER["Spine1"]
    SPINE2_JOINT_INDEX = JOINTS_MAPPER["Spine2"]
    SPINE3_JOINT_INDEX = JOINTS_MAPPER["Spine3"]

    LEFT_COLLAR_JOINT_INDEX = JOINTS_MAPPER["LeftCollar"]
    RIGHT_COLLAR_JOINT_INDEX = JOINTS_MAPPER["RightCollar"]

    JAW_JOINT_INDEX = JOINTS_MAPPER["Jaw"]

    LEFT_CLAVICLE_START_VERTEX_INDEX = 700
    LEFT_CLAVICLE_END_VERTEX_INDEX = 1862

    RIGHT_CLAVICLE_START_VERTEX_INDEX = 4187
    RIGHT_CLAVICLE_END_VERTEX_INDEX = 5325

    LEFT_SCAPULA_ONE_VERTEX_INDEX = 1818
    LEFT_SCAPULA_TWO_VERTEX_INDEX = 2893
    LEFT_SCAPULA_THREE_VERTEX_INDEX = 2937

    RIGHT_SCAPULA_ONE_VERTEX_INDEX = 5279
    RIGHT_SCAPULA_TWO_VERTEX_INDEX = 6352
    RIGHT_SCAPULA_THREE_VERTEX_INDEX = 6394

    LEFT_ANTERIOR_AXILLARY_LINE_START_VERTEX_INDEX = 1286
    LEFT_ANTERIOR_AXILLARY_LINE_END_VERTEX_INDEX = 618

    RIGHT_ANTERIOR_AXILLARY_LINE_START_VERTEX_INDEX = 4892
    RIGHT_ANTERIOR_AXILLARY_LINE_END_VERTEX_INDEX = 4106

    def __init__(self, faces: np.array, vertices: np.array, joints: np.array,
                 camera_translation: np.array, focal_length: np.array,
                 image_size: np.array, fpd_center: np.array, pixel_spacing: float):
        self.faces: np.array = faces
        self.vertices: np.array = vertices
        self.joints: np.array = joints
        self.camera_translation: np.array = camera_translation
        self.focal_length: np.array = focal_length
        self.image_size: np.array = image_size
        self.fpd_center: np.array = fpd_center
        self.pixel_spacing = pixel_spacing

        self._all_limbs: Optional[Limb] = None
        self._left_clavicle: Optional[Limb] = None
        self._right_clavicle: Optional[Limb] = None
        self._left_scapula: Optional[Limb] = None
        self._right_scapula: Optional[Limb] = None
        self._left_upper_limb: Optional[Limb] = None
        self._right_upper_limb: Optional[Limb] = None
        self._left_lower_limb: Optional[Limb] = None
        self._right_lower_limb: Optional[Limb] = None
        self._shoulders_limb: Optional[Limb] = None
        self._left_shoulder_collar_limb: Optional[Limb] = None
        self._right_shoulder_collar_limb: Optional[Limb] = None
        self._whole_spine: Optional[Limb] = None
        self._t_spine: Optional[Limb] = None
        self._left_shoulder_limb: Optional[Limb] = None
        self._right_shoulder_limb: Optional[Limb] = None
        self._left_anterior_axillary_line: Optional[Limb] = None
        self._right_anterior_axillary_line: Optional[Limb] = None

    @property
    def all_limbs(self):
        if self._all_limbs is None:
            self._all_limbs = Limb(
                self.joints[:25],
                np.array([
                    [
                        self.joints[self.PELVIS_JOINT_INDEX],
                        self.joints[self.NECK_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.NECK_JOINT_INDEX],
                        self.joints[self.JAW_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.NECK_JOINT_INDEX],
                        self.joints[self.LEFT_SHOULDER_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                        self.joints[self.LEFT_ELBOW_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_ELBOW_JOINT_INDEX],
                        self.joints[self.LEFT_WRIST_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_WRIST_JOINT_INDEX],
                        self.joints[self.LEFT_HAND_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.NECK_JOINT_INDEX],
                        self.joints[self.RIGHT_SHOULDER_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_SHOULDER_JOINT_INDEX],
                        self.joints[self.RIGHT_ELBOW_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_ELBOW_JOINT_INDEX],
                        self.joints[self.RIGHT_WRIST_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_WRIST_JOINT_INDEX],
                        self.joints[self.RIGHT_HAND_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.PELVIS_JOINT_INDEX],
                        self.joints[self.LEFT_HIP_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_HIP_JOINT_INDEX],
                        self.joints[self.LEFT_KNEE_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_KNEE_JOINT_INDEX],
                        self.joints[self.LEFT_ANKLE_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_ANKLE_JOINT_INDEX],
                        self.joints[self.LEFT_FOOT_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.PELVIS_JOINT_INDEX],
                        self.joints[self.RIGHT_HIP_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_HIP_JOINT_INDEX],
                        self.joints[self.RIGHT_KNEE_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_KNEE_JOINT_INDEX],
                        self.joints[self.RIGHT_ANKLE_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_ANKLE_JOINT_INDEX],
                        self.joints[self.RIGHT_FOOT_JOINT_INDEX]
                    ],
                ])
            )
        return self._all_limbs

    @property
    def left_clavicle(self):
        if self._left_clavicle is None:
            self._left_clavicle = Limb(
                np.array([
                    self.left_clavicle_start,
                    self.left_clavicle_end
                ]),
                np.array([
                    [
                        self.left_clavicle_start,
                        self.left_clavicle_end
                    ]
                ])
            )
        return self._left_clavicle

    @property
    def right_clavicle(self):
        if self._right_clavicle is None:
            self._right_clavicle = Limb(
                np.array([
                    self.right_clavicle_start,
                    self.right_clavicle_end
                ]),
                np.array([
                    [
                        self.right_clavicle_start,
                        self.right_clavicle_end
                    ]
                ])
            )
        return self._right_clavicle

    @property
    def left_scapula(self):
        if self._left_scapula is None:
            self._left_scapula = Limb(
                np.array([
                    self.left_scapula_one,
                    self.left_scapula_two,
                    self.left_scapula_three
                ]),
                np.array([
                    [
                        self.left_scapula_one,
                        self.left_scapula_two
                    ],
                    [
                        self.left_scapula_two,
                        self.left_scapula_three
                    ],
                    [
                        self.left_scapula_three,
                        self.left_scapula_one
                    ]
                ])
            )
        return self._left_scapula

    @property
    def right_scapula(self):
        if self._right_scapula is None:
            self._right_scapula = Limb(
                np.array([
                    self.right_scapula_one,
                    self.right_scapula_two,
                    self.right_scapula_three
                ]),
                np.array([
                    [
                        self.right_scapula_one,
                        self.right_scapula_two
                    ],
                    [
                        self.right_scapula_two,
                        self.right_scapula_three
                    ],
                    [
                        self.right_scapula_three,
                        self.right_scapula_one
                    ]
                ])
            )
        return self._right_scapula

    @property
    def left_upper_limb(self):
        if self._left_upper_limb is None:
            self._left_upper_limb = Limb(
                np.array([
                    self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                    self.joints[self.LEFT_ELBOW_JOINT_INDEX],
                    self.joints[self.LEFT_WRIST_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                        self.joints[self.LEFT_ELBOW_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_ELBOW_JOINT_INDEX],
                        self.joints[self.LEFT_WRIST_JOINT_INDEX]
                    ]
                ])
            )
        return self._left_upper_limb

    @property
    def right_upper_limb(self):
        if self._right_upper_limb is None:
            self._right_upper_limb = Limb(
                np.array([
                    self.joints[self.RIGHT_SHOULDER_JOINT_INDEX],
                    self.joints[self.RIGHT_ELBOW_JOINT_INDEX],
                    self.joints[self.RIGHT_WRIST_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.RIGHT_SHOULDER_JOINT_INDEX],
                        self.joints[self.RIGHT_ELBOW_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_ELBOW_JOINT_INDEX],
                        self.joints[self.RIGHT_WRIST_JOINT_INDEX]
                    ]
                ])
            )
        return self._right_upper_limb

    @property
    def left_lower_limb(self):
        if self._left_lower_limb is None:
            self._left_lower_limb = Limb(
                np.array([
                    self.joints[self.LEFT_HIP_JOINT_INDEX],
                    self.joints[self.LEFT_KNEE_JOINT_INDEX],
                    self.joints[self.LEFT_ANKLE_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.LEFT_HIP_JOINT_INDEX],
                        self.joints[self.LEFT_KNEE_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_KNEE_JOINT_INDEX],
                        self.joints[self.LEFT_ANKLE_JOINT_INDEX]
                    ]
                ])
            )
        return self._left_lower_limb

    @property
    def right_lower_limb(self):
        if self._right_lower_limb is None:
            self._right_lower_limb = Limb(
                np.array([
                    self.joints[self.RIGHT_HIP_JOINT_INDEX],
                    self.joints[self.RIGHT_KNEE_JOINT_INDEX],
                    self.joints[self.RIGHT_ANKLE_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.RIGHT_HIP_JOINT_INDEX],
                        self.joints[self.RIGHT_KNEE_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_KNEE_JOINT_INDEX],
                        self.joints[self.RIGHT_ANKLE_JOINT_INDEX]
                    ]
                ])
            )
        return self._right_lower_limb

    @property
    def shoulders_limb(self):
        if self._shoulders_limb is None:
            self._shoulders_limb = Limb(
                np.array([
                    self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                    self.joints[self.RIGHT_SHOULDER_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                        self.joints[self.RIGHT_SHOULDER_JOINT_INDEX]
                    ]
                ])
            )
        return self._shoulders_limb

    @property
    def left_shoulder_collar_limb(self):
        if self._left_shoulder_collar_limb is None:
            self._left_shoulder_collar_limb = Limb(
                np.array([
                    self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                    self.joints[self.LEFT_COLLAR_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                        self.joints[self.LEFT_COLLAR_JOINT_INDEX]
                    ]
                ])
            )
        return self._left_shoulder_collar_limb

    @property
    def right_shoulder_collar_limb(self):
        if self._right_shoulder_collar_limb is None:
            self._right_shoulder_collar_limb = Limb(
                np.array([
                    self.joints[self.RIGHT_SHOULDER_JOINT_INDEX],
                    self.joints[self.RIGHT_COLLAR_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.RIGHT_SHOULDER_JOINT_INDEX],
                        self.joints[self.RIGHT_COLLAR_JOINT_INDEX]
                    ]
                ])
            )
        return self._right_shoulder_collar_limb

    @property
    def whole_spine(self):
        if self._whole_spine is None:
            self._whole_spine = Limb(
                np.array([
                    self.joints[self.PELVIS_JOINT_INDEX],
                    self.joints[self.SPINE1_JOINT_INDEX],
                    self.joints[self.SPINE2_JOINT_INDEX],
                    self.joints[self.NECK_JOINT_INDEX],
                    self.joints[self.JAW_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.PELVIS_JOINT_INDEX],
                        self.joints[self.SPINE1_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.SPINE1_JOINT_INDEX],
                        self.joints[self.SPINE2_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.SPINE2_JOINT_INDEX],
                        self.joints[self.NECK_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.NECK_JOINT_INDEX],
                        self.joints[self.JAW_JOINT_INDEX]
                    ]
                ])
            )
        return self._whole_spine

    @property
    def t_spine(self):
        if self._t_spine is None:
            ts_ratio = [2, 3, 3, 4, 4, 6, 6, 6, 6, 6, 7, 6]
            all_ts = sum(ts_ratio)
            t_unit_length = (self.spine1_joint - self.neck_joint) / all_ts
            t_spine_points = np.array([self.neck_joint + t_unit_length * sum(ts_ratio[:i + 1])
                                       for i in range(12)])
            t_spine_links = np.array([
                [
                    self.neck_joint + t_unit_length * sum(ts_ratio[:i + 1]),
                    self.neck_joint + t_unit_length * sum(ts_ratio[:i + 2])
                ] for i in range(11)])

            self._t_spine = Limb(
                t_spine_points,
                t_spine_links
            )
        return self._t_spine

    @property
    def left_shoulder_limb(self):
        if self._left_shoulder_limb is None:
            self._left_shoulder_limb = Limb(
                np.array([
                    self.joints[self.NECK_JOINT_INDEX],
                    self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                    self.joints[self.LEFT_ELBOW_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.NECK_JOINT_INDEX],
                        self.joints[self.LEFT_SHOULDER_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.LEFT_SHOULDER_JOINT_INDEX],
                        self.joints[self.LEFT_ELBOW_JOINT_INDEX]
                    ]
                ])
            )
        return self._left_shoulder_limb

    @property
    def right_shoulder_limb(self):
        if self._right_shoulder_limb is None:
            self._right_shoulder_limb = Limb(
                np.array([
                    self.joints[self.NECK_JOINT_INDEX],
                    self.joints[self.RIGHT_SHOULDER_JOINT_INDEX],
                    self.joints[self.RIGHT_ELBOW_JOINT_INDEX]
                ]),
                np.array([
                    [
                        self.joints[self.NECK_JOINT_INDEX],
                        self.joints[self.RIGHT_SHOULDER_JOINT_INDEX]
                    ],
                    [
                        self.joints[self.RIGHT_SHOULDER_JOINT_INDEX],
                        self.joints[self.RIGHT_ELBOW_JOINT_INDEX]
                    ]
                ])
            )
        return self._right_shoulder_limb

    @property
    def left_anterior_axillary_line(self) -> Limb:
        if self._left_anterior_axillary_line is None:
            self._left_anterior_axillary_line = Limb(
                np.array([
                    self.left_shoulder_joint,
                    self.left_anterior_axillary_line_end
                ]),
                np.array([
                    [
                        self.left_shoulder_joint,
                        self.left_anterior_axillary_line_end
                    ]
                ])
            )
        return self._left_anterior_axillary_line

    @property
    def right_anterior_axillary_line(self) -> Limb:
        if self._right_anterior_axillary_line is None:
            self._right_anterior_axillary_line = Limb(
                np.array([
                    self.right_shoulder_joint,
                    self.right_anterior_axillary_line_end
                ]),
                np.array([
                    [
                        self.right_shoulder_joint,
                        self.right_anterior_axillary_line_end
                    ]
                ])
            )
        return self._right_anterior_axillary_line

    @property
    def neck_joint(self) -> np.array:
        return self.joints[self.NECK_JOINT_INDEX]

    @property
    def spine1_joint(self) -> np.array:
        return self.joints[self.SPINE1_JOINT_INDEX]

    @property
    def left_shoulder_joint(self) -> np.array:
        return self.joints[self.LEFT_SHOULDER_JOINT_INDEX]

    @property
    def left_elbow_joint(self) -> np.array:
        return self.joints[self.LEFT_ELBOW_JOINT_INDEX]

    @property
    def left_wrist_joint(self) -> np.array:
        return self.joints[self.LEFT_WRIST_JOINT_INDEX]

    @property
    def right_shoulder_joint(self) -> np.array:
        return self.joints[self.RIGHT_SHOULDER_JOINT_INDEX]

    @property
    def right_elbow_joint(self) -> np.array:
        return self.joints[self.RIGHT_ELBOW_JOINT_INDEX]

    @property
    def right_wrist_joint(self) -> np.array:
        return self.joints[self.RIGHT_WRIST_JOINT_INDEX]

    @property
    def left_collar_joint(self) -> np.array:
        return self.joints[self.LEFT_COLLAR_JOINT_INDEX]

    @property
    def right_collar_joint(self) -> np.array:
        return self.joints[self.RIGHT_COLLAR_JOINT_INDEX]

    @property
    def left_clavicle_start(self) -> np.array:
        return self.vertices[self.LEFT_CLAVICLE_START_VERTEX_INDEX]

    @property
    def right_clavicle_start(self) -> np.array:
        return self.vertices[self.RIGHT_CLAVICLE_START_VERTEX_INDEX]

    @property
    def left_clavicle_end(self) -> np.array:
        return self.vertices[self.LEFT_CLAVICLE_END_VERTEX_INDEX]

    @property
    def right_clavicle_end(self) -> np.array:
        return self.vertices[self.RIGHT_CLAVICLE_END_VERTEX_INDEX]

    @property
    def left_scapula_one(self) -> np.array:
        return self.vertices[self.LEFT_SCAPULA_ONE_VERTEX_INDEX]

    @property
    def left_scapula_two(self) -> np.array:
        return self.vertices[self.LEFT_SCAPULA_TWO_VERTEX_INDEX]

    @property
    def left_scapula_three(self) -> np.array:
        return self.vertices[self.LEFT_SCAPULA_THREE_VERTEX_INDEX]

    @property
    def right_scapula_one(self) -> np.array:
        return self.vertices[self.RIGHT_SCAPULA_ONE_VERTEX_INDEX]

    @property
    def right_scapula_two(self) -> np.array:
        return self.vertices[self.RIGHT_SCAPULA_TWO_VERTEX_INDEX]

    @property
    def right_scapula_three(self) -> np.array:
        return self.vertices[self.RIGHT_SCAPULA_THREE_VERTEX_INDEX]

    @property
    def t5(self) -> np.array:
        return self.t_spine.joints[4]

    @property
    def t6(self) -> np.array:
        return self.t_spine.joints[5]

    @property
    def t7(self) -> np.array:
        return self.t_spine.joints[6]

    @property
    def t6_center_uv(self) -> np.array:
        return xyz2uv(0.5 * np.array([self.t5 + self.t6]),
                      self.camera_translation, self.focal_length / 256 * np.max(self.image_size),
                      np.array((self.image_size[1] / 2., self.image_size[0] / 2.)))

    @property
    def t7_uv(self) -> np.array:
        return xyz2uv(np.array([self.t7]),
                      self.camera_translation, self.focal_length / 256 * np.max(self.image_size),
                      np.array((self.image_size[1] / 2., self.image_size[0] / 2.)))

    @property
    def fpd_center_uv(self) -> np.array:
        return np.array([self.fpd_center])

    @property
    def left_anterior_axillary_line_start(self) -> np.array:
        return self.vertices[self.LEFT_ANTERIOR_AXILLARY_LINE_START_VERTEX_INDEX]

    @property
    def right_anterior_axillary_line_start(self) -> np.array:
        return self.vertices[self.RIGHT_ANTERIOR_AXILLARY_LINE_START_VERTEX_INDEX]

    @property
    def left_anterior_axillary_line_end(self) -> np.array:
        return self.vertices[self.LEFT_ANTERIOR_AXILLARY_LINE_END_VERTEX_INDEX]

    @property
    def right_anterior_axillary_line_end(self) -> np.array:
        return self.vertices[self.RIGHT_ANTERIOR_AXILLARY_LINE_END_VERTEX_INDEX]

    @property
    def left_anterior_axillary_line_start_uv(self) -> np.array:
        return xyz2uv(np.array([self.left_anterior_axillary_line_start]),
                      self.camera_translation, self.focal_length / 256 * np.max(self.image_size),
                      np.array((self.image_size[1] / 2., self.image_size[0] / 2.)))

    @property
    def right_anterior_axillary_line_start_uv(self) -> np.array:
        return xyz2uv(np.array([self.right_anterior_axillary_line_start]),
                      self.camera_translation, self.focal_length / 256 * np.max(self.image_size),
                      np.array((self.image_size[1] / 2., self.image_size[0] / 2.)))

    @property
    def left_anterior_axillary_line_end_uv(self) -> np.array:
        return xyz2uv(np.array([self.left_anterior_axillary_line_end]),
                      self.camera_translation, self.focal_length / 256 * np.max(self.image_size),
                      np.array((self.image_size[1] / 2., self.image_size[0] / 2.)))

    @property
    def right_anterior_axillary_line_end_uv(self) -> np.array:
        return xyz2uv(np.array([self.right_anterior_axillary_line_end]),
                      self.camera_translation, self.focal_length / 256 * np.max(self.image_size),
                      np.array((self.image_size[1] / 2., self.image_size[0] / 2.)))

    @property
    def anterior_axillary_line_uv(self) -> np.array:
        return np.array([
            self.left_anterior_axillary_line_start_uv,
            self.left_anterior_axillary_line_end_uv
        ])

    @property
    def t6_horn_line_uv(self) -> np.array:
        return np.array([self.t7_uv, self.t7_uv + np.array([1, 0])])

    @property
    def t7_anterior_axillary_line_perpendicular_uv(self) -> np.array:
        return np.array([perpendicular(self.anterior_axillary_line_uv, self.t7_uv)])
