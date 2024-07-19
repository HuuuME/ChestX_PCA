from typing import Optional

import numpy as np


class VisibleTypes:

    MESH = 0
    POINTS = 1
    LINES = 2
    PIXELS = 3


class Visible:

    def __init__(self, visible_type: VisibleTypes,
                 vertices: np.array = None, faces: np.array = None,
                 points: np.array = None, lines: np.array = None,
                 pixels: np.array = None,
                 color: tuple = (0.35, 0.35, 0.35, 0.5)):
        self.visible_type = visible_type
        self.vertices: Optional[np.array] = vertices
        self.faces: Optional[np.array] = faces
        self.points: Optional[np.array] = points
        self.lines: Optional[np.array] = lines
        self.pixels: Optional[np.array] = pixels
        self.color = color
