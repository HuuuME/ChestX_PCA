import numpy as np

from model.digital_man import DigitalMan
from model.evaluator import Evaluator

from model.visible import Visible, VisibleTypes


class CriterionBase:

    name = "base"

    def __init__(self, dm: DigitalMan):
        self.dm = dm
        self.evaluator = Evaluator(dm)

    @property
    def report(self):
        raise NotImplemented

    @property
    def visible(self):
        raise NotImplemented


class DummyCriterion(CriterionBase):

    name = "height"

    @property
    def report(self):
        return self.evaluator.height

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.8))


class LeftClavicleRollAngleCriterion(CriterionBase):

    name = "left clavicle roll angle"

    @property
    def report(self):
        return self.evaluator.left_clavicle_roll_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_clavicle.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_clavicle.links, color=(0, 0, 1, 1))


class RightClavicleRollAngleCriterion(CriterionBase):

    name = "right clavicle roll angle"

    @property
    def report(self):
        return self.evaluator.right_clavicle_roll_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_clavicle.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_clavicle.links, color=(0, 0, 1, 1))


class LeftClavicleYawAngleCriterion(CriterionBase):

    name = "left clavicle yaw angle"

    @property
    def report(self):
        return self.evaluator.left_clavicle_yaw_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_clavicle.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_clavicle.links, color=(0, 0, 1, 1))


class RightClavicleYawAngleCriterion(CriterionBase):

    name = "right clavicle yaw angle"

    @property
    def report(self):
        return self.evaluator.right_clavicle_yaw_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_clavicle.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_clavicle.links, color=(0, 0, 1, 1))


class LeftScapulaPitchAngleCriterion(CriterionBase):

    name = "left scapula pitch angle"

    @property
    def report(self):
        return self.evaluator.left_scapula_pitch_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_scapula.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_scapula.links, color=(0, 0, 1, 1))


class RightScapulaPitchAngleCriterion(CriterionBase):

    name = "right scapula pitch angle"

    @property
    def report(self):
        return self.evaluator.right_scapula_pitch_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_scapula.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_scapula.links, color=(0, 0, 1, 1))


class LeftScapulaYawAngleCriterion(CriterionBase):

    name = "left scapula yaw angle"

    @property
    def report(self):
        return self.evaluator.left_scapula_yaw_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_scapula.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_scapula.links, color=(0, 0, 1, 1))


class RightScapulaYawAngleCriterion(CriterionBase):

    name = "right scapula yaw angle"

    @property
    def report(self):
        return self.evaluator.right_scapula_yaw_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_scapula.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_scapula.links, color=(0, 0, 1, 1))


class LeftScapulaRollAngleCriterion(CriterionBase):

    name = "left scapula roll angle"

    @property
    def report(self):
        return self.evaluator.left_scapula_roll_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_scapula.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_scapula.links, color=(0, 0, 1, 1))


class RightScapulaRollAngleCriterion(CriterionBase):

    name = "right scapula roll angle"

    @property
    def report(self):
        return self.evaluator.right_scapula_roll_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_scapula.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_scapula.links, color=(0, 0, 1, 1))


class LeftCollarShoulderPitchAngleCriterion(CriterionBase):

    name = "left collar shoulder pitch angle"

    @property
    def report(self):
        return self.evaluator.left_collar_shoulder_pitch_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_shoulder_collar_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_shoulder_collar_limb.links, color=(0, 0, 1, 1))


class RightCollarShoulderPitchAngleCriterion(CriterionBase):

    name = "right collar shoulder pitch angle"

    @property
    def report(self):
        return self.evaluator.right_collar_shoulder_pitch_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_shoulder_collar_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_shoulder_collar_limb.links, color=(0, 0, 1, 1))


class LeftCollarShoulderYawAngleCriterion(CriterionBase):

    name = "left collar shoulder yaw angle"

    @property
    def report(self):
        return self.evaluator.left_collar_shoulder_yaw_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_shoulder_collar_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_shoulder_collar_limb.links, color=(0, 0, 1, 1))


class RightCollarShoulderYawAngleCriterion(CriterionBase):

    name = "right collar shoulder yaw angle"

    @property
    def report(self):
        return self.evaluator.right_collar_shoulder_yaw_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_shoulder_collar_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_shoulder_collar_limb.links, color=(0, 0, 1, 1))


class LeftShoulderAngleCriterion(CriterionBase):

    name = "left shoulder angle"

    @property
    def report(self):
        return self.evaluator.left_shoulder_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_shoulder_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_shoulder_limb.links, color=(0, 0, 1, 1))


class RightShoulderAngleCriterion(CriterionBase):

    name = "right shoulder angle"

    @property
    def report(self):
        return self.evaluator.right_shoulder_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_shoulder_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_shoulder_limb.links, color=(0, 0, 1, 1))


class TSpineRollAngleCriterion(CriterionBase):

    name = "t-spine roll angle"

    @property
    def report(self):
        return self.evaluator.t_spine_roll_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.t_spine.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.t_spine.links, color=(0, 0, 1, 1))


class ShouldersPitchAngleCriterion(CriterionBase):

    name = "shoulders pitch angle"

    @property
    def report(self):
        return self.evaluator.shoulders_pitch_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.shoulders_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.shoulders_limb.links, color=(0, 0, 1, 1))


class ShouldersYawAngleCriterion(CriterionBase):

    name = "shoulders yaw angle"

    @property
    def report(self):
        return self.evaluator.shoulders_yaw_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.shoulders_limb.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.shoulders_limb.links, color=(0, 0, 1, 1))


class PAPositionOffsetXCriterion(CriterionBase):

    name = "PA position offset x"

    @property
    def report(self):
        return self.evaluator.pa_position_offset_x

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-np.array([self.dm.t5]), color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.POINTS, points=-np.array([self.dm.t6]), color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.t6_center_uv, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.fpd_center_uv, color=(1, 0, 0, 1))


class PAPositionOffsetYCriterion(CriterionBase):

    name = "PA position offset y"

    @property
    def report(self):
        return self.evaluator.pa_position_offset_y

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-np.array([self.dm.t5]), color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.POINTS, points=-np.array([self.dm.t6]), color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.t6_center_uv, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.fpd_center_uv, color=(1, 0, 0, 1))


class LeftAnteriorAxillaryLineRollAngleCriterion(CriterionBase):

    name = "left anterior axillary line roll angle"

    @property
    def report(self):
        return self.evaluator.left_anterior_axillary_line_roll_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.left_anterior_axillary_line.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_anterior_axillary_line.links, color=(0, 0, 1, 1))


class RightAnteriorAxillaryLineRollAngleCriterion(CriterionBase):

    name = "right anterior axillary line roll angle"

    @property
    def report(self):
        return self.evaluator.right_anterior_axillary_line_roll_angle

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.POINTS, points=-self.dm.right_anterior_axillary_line.joints, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.right_anterior_axillary_line.links, color=(0, 0, 1, 1))


class LAPositionOffsetXCriterion(CriterionBase):

    name = "LA position offset x"

    @property
    def report(self):
        return self.evaluator.la_position_offset_x

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_anterior_axillary_line.links, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.t7_anterior_axillary_line_perpendicular_uv,
                      color=(0, 1, 0, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.fpd_center_uv, color=(1, 0, 0, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.t7_uv, color=(1, 1, 0, 1))


class LAPositionOffsetYCriterion(CriterionBase):

    name = "LA position offset y"

    @property
    def report(self):
        return self.evaluator.la_position_offset_y

    @property
    def visible(self):
        yield Visible(VisibleTypes.MESH, vertices=-self.dm.vertices, faces=self.dm.faces, color=(1, 1, 0.9, 0.5))
        yield Visible(VisibleTypes.LINES, lines=-self.dm.left_anterior_axillary_line.links, color=(0, 0, 1, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.t7_anterior_axillary_line_perpendicular_uv,
                      color=(0, 1, 0, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.fpd_center_uv, color=(1, 0, 0, 1))
        yield Visible(VisibleTypes.PIXELS, pixels=self.dm.t7_uv, color=(1, 1, 0, 1))
