import numpy as np

from model.digital_man import DigitalMan


class Evaluator:

    def __init__(self, dm: DigitalMan):
        self.dm = dm

        self._height = None
        self._left_clavicle_roll_angle = None
        self._right_clavicle_roll_angle = None
        self._left_clavicle_yaw_angle = None
        self._right_clavicle_yaw_angle = None
        self._left_scapula_pitch_angle = None
        self._right_scapula_pitch_angle = None
        self._left_scapula_yaw_angle = None
        self._right_scapula_yaw_angle = None
        self._left_scapula_roll_angle = None
        self._right_scapula_roll_angle = None
        self._t_spine_pitch_angle = None
        self._t_spine_yaw_angle = None
        self._t_spine_roll_angle = None
        self._shoulders_yaw_angle = None
        self._shoulders_pitch_angle = None
        self._left_elbow_angle = None
        self._right_elbow_angle = None
        self._left_collar_shoulder_pitch_angle = None
        self._right_collar_shoulder_pitch_angle = None
        self._left_collar_shoulder_yaw_angle = None
        self._right_collar_shoulder_yaw_angle = None
        self._left_shoulder_angle = None
        self._right_shoulder_angle = None
        self._pa_position_offset_x = None
        self._pa_position_offset_y = None
        self._la_position_offset_x = None
        self._la_position_offset_y = None
        self._left_anterior_axillary_line_roll_angle = None
        self._right_anterior_axillary_line_roll_angle = None

    @property
    def height(self):
        if self._height is None:
            top = np.max(self.dm.vertices, axis=0)[1]
            bottom = np.min(self.dm.vertices, axis=0)[1]
            self._height = top - bottom
        return self._height

    @property
    def left_clavicle_roll_angle(self):
        if self._left_clavicle_roll_angle is None:
            line = self.dm.left_shoulder_joint - self.dm.left_collar_joint
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._left_clavicle_roll_angle = 180 - angle
            elif angle < -90:
                self._left_clavicle_roll_angle = -180 - angle
            else:
                self._left_clavicle_roll_angle = angle
        return self._left_clavicle_roll_angle

    @property
    def right_clavicle_roll_angle(self):
        if self._right_clavicle_roll_angle is None:
            line = self.dm.right_shoulder_joint - self.dm.right_collar_joint
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._right_clavicle_roll_angle = 180 - angle
            elif angle < -90:
                self._right_clavicle_roll_angle = -180 - angle
            else:
                self._right_clavicle_roll_angle = angle
        return self._right_clavicle_roll_angle

    @property
    def left_clavicle_yaw_angle(self):
        if self._left_clavicle_yaw_angle is None:
            line = self.dm.left_shoulder_joint - self.dm.left_collar_joint
            angle = np.arctan(line[2] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._left_clavicle_yaw_angle = 180 - angle
            elif angle < -90:
                self._left_clavicle_yaw_angle = -180 - angle
            else:
                self._left_clavicle_yaw_angle = angle
        return self._left_clavicle_yaw_angle

    @property
    def right_clavicle_yaw_angle(self):
        if self._right_clavicle_yaw_angle is None:
            line = self.dm.right_shoulder_joint - self.dm.right_collar_joint
            angle = np.arctan(line[2] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._right_clavicle_yaw_angle = 180 - angle
            elif angle < -90:
                self._right_clavicle_yaw_angle = -180 - angle
            else:
                self._right_clavicle_yaw_angle = angle
        return self._right_clavicle_yaw_angle

    @property
    def left_scapula_pitch_angle(self):
        if self._left_scapula_pitch_angle is None:
            line = self.dm.left_scapula_one - self.dm.left_scapula_two
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._left_scapula_pitch_angle = 180 - angle
            elif angle < -90:
                self._left_scapula_pitch_angle = -180 - angle
            else:
                self._left_scapula_pitch_angle = angle
        return self._left_scapula_pitch_angle

    @property
    def right_scapula_pitch_angle(self):
        if self._right_scapula_pitch_angle is None:
            line = self.dm.right_scapula_one - self.dm.right_scapula_two
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._right_scapula_pitch_angle = 180 - angle
            elif angle < -90:
                self._right_scapula_pitch_angle = -180 - angle
            else:
                self._right_scapula_pitch_angle = angle
        return self._right_scapula_pitch_angle

    @property
    def left_scapula_yaw_angle(self):
        if self._left_scapula_yaw_angle is None:
            line = self.dm.left_scapula_one - self.dm.left_scapula_two
            angle = np.arctan(line[2] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._left_scapula_yaw_angle = 180 - angle
            elif angle < -90:
                self._left_scapula_yaw_angle = -180 - angle
            else:
                self._left_scapula_yaw_angle = angle
        return self._left_scapula_yaw_angle

    @property
    def right_scapula_yaw_angle(self):
        if self._right_scapula_yaw_angle is None:
            line = self.dm.right_scapula_one - self.dm.right_scapula_two
            angle = np.arctan(line[2] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._right_scapula_yaw_angle = 180 - angle
            elif angle < -90:
                self._right_scapula_yaw_angle = -180 - angle
            else:
                self._right_scapula_yaw_angle = angle
        return self._right_scapula_yaw_angle

    @property
    def left_scapula_roll_angle(self):
        if self._left_scapula_roll_angle is None:
            line = self.dm.left_scapula_one - self.dm.left_scapula_three
            angle = np.arctan(line[0] / line[1])
            self._left_scapula_roll_angle = angle / np.pi * 180
        return self._left_scapula_roll_angle

    @property
    def right_scapula_roll_angle(self):
        if self._right_scapula_roll_angle is None:
            line = self.dm.right_scapula_one - self.dm.right_scapula_three
            angle = np.arctan(line[0] / line[1])
            self._right_scapula_roll_angle = angle / np.pi * 180
        return self._right_scapula_roll_angle

    @property
    def left_elbow_angle(self):
        if self._left_elbow_angle is None:
            l1 = self.dm.left_shoulder_joint - self.dm.left_elbow_joint
            l2 = self.dm.left_wrist_joint - self.dm.left_elbow_joint
            angle = np.arccos(np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2)))
            self._left_elbow_angle = angle / np.pi * 180
        return self._left_elbow_angle

    @property
    def right_elbow_angle(self):
        if self._right_elbow_angle is None:
            l1 = self.dm.right_shoulder_joint - self.dm.right_elbow_joint
            l2 = self.dm.right_wrist_joint - self.dm.right_elbow_joint
            angle = np.arccos(np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2)))
            self._right_elbow_angle = angle / np.pi * 180
        return self._right_elbow_angle

    @property
    def left_collar_shoulder_pitch_angle(self):
        if self._left_collar_shoulder_pitch_angle is None:
            line = self.dm.left_collar_joint - self.dm.left_shoulder_joint
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._left_collar_shoulder_pitch_angle = 180 - angle
            elif angle < -90:
                self._left_collar_shoulder_pitch_angle = -180 - angle
            else:
                self._left_collar_shoulder_pitch_angle = angle
        return self._left_collar_shoulder_pitch_angle

    @property
    def right_collar_shoulder_pitch_angle(self):
        if self._right_collar_shoulder_pitch_angle is None:
            line = self.dm.right_collar_joint - self.dm.right_shoulder_joint
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._right_collar_shoulder_pitch_angle = 180 - angle
            elif angle < -90:
                self._right_collar_shoulder_pitch_angle = -180 - angle
            else:
                self._right_collar_shoulder_pitch_angle = angle
        return self._right_collar_shoulder_pitch_angle

    @property
    def left_collar_shoulder_yaw_angle(self):
        if self._left_collar_shoulder_yaw_angle is None:
            line = self.dm.left_collar_joint - self.dm.left_shoulder_joint
            angle = np.arctan(line[2] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._left_collar_shoulder_yaw_angle = 180 - angle
            elif angle < -90:
                self._left_collar_shoulder_yaw_angle = -180 - angle
            else:
                self._left_collar_shoulder_yaw_angle = angle
        return self._left_collar_shoulder_yaw_angle

    @property
    def right_collar_shoulder_yaw_angle(self):
        if self._right_collar_shoulder_yaw_angle is None:
            line = self.dm.right_collar_joint - self.dm.right_shoulder_joint
            angle = np.arctan(line[2] / line[0])
            angle = angle / np.pi * 180
            angle = np.abs(angle)
            if angle > 90:
                self._right_collar_shoulder_yaw_angle = 180 - angle
            elif angle < -90:
                self._right_collar_shoulder_yaw_angle = -180 - angle
            else:
                self._right_collar_shoulder_yaw_angle = angle
        return self._right_collar_shoulder_yaw_angle

    @property
    def left_shoulder_angle(self):
        if self._left_shoulder_angle is None:
            # l1 = self.dm.neck_joint - self.dm.left_shoulder_joint
            # l2 = self.dm.left_shoulder_joint - self.dm.left_elbow_joint
            # angle = np.arccos(np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2)))
            # self._left_shoulder_angle = angle / np.pi * 180
            line = self.dm.left_shoulder_joint - self.dm.left_elbow_joint
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            self._left_shoulder_angle = np.abs(angle)
        return self._left_shoulder_angle

    @property
    def right_shoulder_angle(self):
        if self._right_shoulder_angle is None:
            # l1 = self.dm.neck_joint - self.dm.right_shoulder_joint
            # l2 = self.dm.right_shoulder_joint - self.dm.right_elbow_joint
            # angle = np.arccos(np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2)))
            # self._right_shoulder_angle = angle / np.pi * 180
            line = self.dm.right_shoulder_joint - self.dm.right_elbow_joint
            angle = np.arctan(line[1] / line[0])
            angle = angle / np.pi * 180
            self._right_shoulder_angle = np.abs(angle)
        return self._right_shoulder_angle

    @property
    def t_spine_pitch_angle(self):
        raise NotImplemented

    @property
    def t_spine_yaw_angle(self):
        raise NotImplemented

    @property
    def t_spine_roll_angle(self):
        if self._t_spine_roll_angle is None:
            line = self.dm.neck_joint - self.dm.spine1_joint
            angle = np.arctan(line[0] / line[1])
            self._t_spine_roll_angle = angle / np.pi * 180
        return self._t_spine_roll_angle

    @property
    def shoulders_pitch_angle(self):
        if self._shoulders_pitch_angle is None:
            line = self.dm.right_shoulder_joint - self.dm.left_shoulder_joint
            angle = np.arctan(line[1] / line[2])
            self._shoulders_pitch_angle = angle / np.pi * 180
        return self._shoulders_pitch_angle

    @property
    def shoulders_yaw_angle(self):
        if self._shoulders_yaw_angle is None:
            line = self.dm.right_shoulder_joint - self.dm.left_shoulder_joint
            angle = np.arctan(line[2] / line[0])
            self._shoulders_yaw_angle = angle / np.pi * 180
        return self._shoulders_yaw_angle

    @property
    def pa_position_offset_x(self):
        if self._pa_position_offset_x is None:
            self._pa_position_offset_x = (self.dm.t6_center_uv[0][0] -
                                          self.dm.fpd_center_uv[0][0]) * self.dm.pixel_spacing
        return self._pa_position_offset_x

    @property
    def pa_position_offset_y(self):
        if self._pa_position_offset_y is None:
            self._pa_position_offset_y = (self.dm.t6_center_uv[0][1] -
                                          self.dm.fpd_center_uv[0][1]) * self.dm.pixel_spacing
        return self._pa_position_offset_y

    @property
    def la_position_offset_x(self):
        if self._la_position_offset_x is None:
            self._pa_position_offset_x = (self.dm.t7_anterior_axillary_line_perpendicular_uv[0][0] -
                                          self.dm.fpd_center_uv[0][0]) * self.dm.pixel_spacing
            return self._pa_position_offset_x

    @property
    def la_position_offset_y(self):
        if self._la_position_offset_y is None:
            self._pa_position_offset_y = (self.dm.t7_anterior_axillary_line_perpendicular_uv[0][1] -
                                          self.dm.fpd_center_uv[0][1]) * self.dm.pixel_spacing
            return self._pa_position_offset_y

    @property
    def left_anterior_axillary_line_roll_angle(self):
        if self._left_anterior_axillary_line_roll_angle is None:
            line = self.dm.left_shoulder_joint - self.dm.left_anterior_axillary_line_end
            angle = np.arctan(line[0] / line[1])
            self._left_anterior_axillary_line_roll_angle = angle / np.pi * 180
        return self._left_anterior_axillary_line_roll_angle

    @property
    def right_anterior_axillary_line_roll_angle(self):
        if self._right_anterior_axillary_line_roll_angle is None:
            line = self.dm.right_shoulder_joint - self.dm.right_anterior_axillary_line_end
            angle = np.arctan(line[0] / line[1])
            self._right_anterior_axillary_line_roll_angle = angle / np.pi * 180
        return self._right_anterior_axillary_line_roll_angle
