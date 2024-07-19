import cv2
import numpy as np

from driver.digital_man_loader import load_digital_man

from utils.renderer import Renderer

from model.exam import ExamDummy
from model.exam import ExamChestPA
from model.exam import ExamChestLA

from model.criterions import ShouldersYawAngleCriterion, ShouldersPitchAngleCriterion


def evaluate_dummy_exam(pkl_path: str):
    dm = load_digital_man(pkl_path)

    exam_dummy = ExamDummy()
    renderer = Renderer()
    for criterion in exam_dummy.evaluate(dm):
        out_img = renderer.render(criterion.visible, dm.image_size, dm.camera_translation,
                                  rotation=np.array((0, 180, 0)),
                                  focal_length=dm.focal_length)
        print("{}: {}".format(criterion.name, criterion.report))
        cv2.imshow(criterion.name, out_img)


def evaluate_one_chest_pa_exam(pkl_path: str):
    dm = load_digital_man(pkl_path)

    exam_chest_pa = ExamChestPA()
    renderer = Renderer()
    for criterion in exam_chest_pa.evaluate(dm):
        rotation = np.array((0, 180, 0))
        if isinstance(criterion, ShouldersYawAngleCriterion):
            rotation = np.array((45, 180, 0))
        out_img = renderer.render(list(criterion.visible), dm.image_size, dm.camera_translation,
                                  rotation=rotation,
                                  focal_length=dm.focal_length)
        print("PAChest {}: {}".format(criterion.name, criterion.report))
        cv2.imshow("PAChest {}".format(criterion.name), out_img)


def evaluate_one_chest_la_exam(pkl_path: str):
    dm = load_digital_man(pkl_path)

    exam_chest_la = ExamChestLA()
    renderer = Renderer()
    for criterion in exam_chest_la.evaluate(dm):
        rotation = np.array((0, 180, 0))
        if isinstance(criterion, ShouldersYawAngleCriterion):
            rotation = np.array((45, 180, 0))
        if isinstance(criterion, ShouldersPitchAngleCriterion):
            rotation = np.array((0, 270, 0))
        out_img = renderer.render(list(criterion.visible), dm.image_size, dm.camera_translation,
                                  rotation=rotation,
                                  focal_length=dm.focal_length)
        print("LAChest {}: {}".format(criterion.name, criterion.report))
        cv2.imshow("LAChest {}".format(criterion.name), out_img)


if __name__ == "__main__":
    evaluate_dummy_exam("/home/hume/PycharmProjects/Omnisense/ChestX_PCA/demo/pa_chest_demo_1.pkl")
    evaluate_one_chest_pa_exam("/home/hume/PycharmProjects/Omnisense/ChestX_PCA/demo/pa_chest_demo_1.pkl")
    evaluate_one_chest_la_exam("/home/hume/PycharmProjects/Omnisense/ChestX_PCA/demo/la_chest_demo_1.pkl")
    cv2.waitKey()
