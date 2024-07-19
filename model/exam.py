from typing import Optional, Iterable

from model.digital_man import DigitalMan

from model.criterions import CriterionBase, DummyCriterion, LeftShoulderAngleCriterion, RightShoulderAngleCriterion, \
    LeftAnteriorAxillaryLineRollAngleCriterion, RightAnteriorAxillaryLineRollAngleCriterion, TSpineRollAngleCriterion, \
    LAPositionOffsetXCriterion, LAPositionOffsetYCriterion, ShouldersYawAngleCriterion, ShouldersPitchAngleCriterion,\
    LeftClavicleRollAngleCriterion, RightClavicleRollAngleCriterion, LeftScapulaYawAngleCriterion, \
    RightScapulaYawAngleCriterion, PAPositionOffsetXCriterion, PAPositionOffsetYCriterion


class ExamBase:

    def __init__(self):

        self.standard_dm: Optional[DigitalMan] = None

    def load_standard_dm(self, standard_dm: Optional[DigitalMan]):
        self.standard_dm = standard_dm

    def evaluate(self, dm: DigitalMan) -> Iterable[CriterionBase]:
        raise NotImplemented


class ExamDummy(ExamBase):

    def evaluate(self, dm: DigitalMan):
        yield DummyCriterion(dm)


class ExamChestPA(ExamBase):

    def evaluate(self, dm: DigitalMan):
        yield LeftClavicleRollAngleCriterion(dm)
        yield RightClavicleRollAngleCriterion(dm)
        yield LeftScapulaYawAngleCriterion(dm)
        yield RightScapulaYawAngleCriterion(dm)
        yield TSpineRollAngleCriterion(dm)
        yield ShouldersYawAngleCriterion(dm)


class ExamChestLA(ExamBase):

    def evaluate(self, dm: DigitalMan):
        yield LeftShoulderAngleCriterion(dm)
        yield RightShoulderAngleCriterion(dm)
        yield LeftAnteriorAxillaryLineRollAngleCriterion(dm)
        yield ShouldersYawAngleCriterion(dm)
