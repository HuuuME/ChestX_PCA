from typing import List

import pyrender
import numpy as np
import cv2
import trimesh
import torch

from model.visible import Visible, VisibleTypes


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def resize(image: np.array, height: int, width: int) -> np.array:
    origin_height, origin_width, _ = image.shape
    ratio = min(float(height) / origin_height, float(width) / origin_width)
    resized_height = int(origin_height * ratio + .5)
    resized_width = int(origin_width * ratio + .5)
    resized_image = cv2.resize(image, [resized_width, resized_height])
    top, left = (height - resized_height) // 2, (width - resized_width) // 2
    new_image = np.zeros([height, width, 3])
    new_image[top: resized_height + top, left: resized_width + left] = resized_image
    return top, left, new_image


def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def generate_cube_mesh(points: np.array, n: int, r: float):
    vertices = []
    faces = []

    ang = 2 * np.pi / n
    vs = []
    offset = 0
    for p in points:
        offset += len(vs)
        vs = []
        fs = []
        for i in range(n // 2 + 1):
            xr = np.sin(i * ang) * r
            y = np.cos(i * ang) * r + p[1]
            for j in range(n):
                x = np.sin(j * ang) * xr + p[0]
                z = np.cos(j * ang) * xr + p[2]
                vs.append(np.array([x, y, z]))

                if j == n - 1:
                    x0 = np.sin(0) * xr + p[0]
                    z0 = np.cos(0) * xr + p[2]
                    vs.append(np.array([x0, y, z0]))

                if i < n // 2 and j < n:
                    fs.append(np.array(
                        [
                            i * (n + 1) + j + offset,
                            i * (n + 1) + j + 1 + offset,
                            (i + 1) * (n + 1) + j + offset
                        ]
                    ))
                    fs.append(np.array(
                        [
                            (i + 1) * (n + 1) + j + offset,
                            i * (n + 1) + j + 1 + offset,
                            (i + 1) * (n + 1) + j + 1 + offset
                        ]
                    ))
        vertices.extend(vs)
        faces.extend(fs)
    return vertices, faces


def gen_transform_matrix(center: np.array, direction: np.array):
    trans = trimesh.transformations.rotation_matrix(np.radians(direction[1]), [0, 1, 0])
    trans[:3, 3] = center

    return trans


def transform(mesh: trimesh.Trimesh, center: tuple, direction: tuple):
    norm_x = np.array(direction) / np.linalg.norm(direction)
    norm_z = np.cross(norm_x, [0, 1.0, 0.0000001])
    norm_z /= np.linalg.norm(norm_z)
    norm_y = np.cross(norm_z, norm_x)

    trans = np.zeros((4, 4))
    trans[:3, 0] = norm_x
    trans[:3, 1] = norm_y
    trans[:3, 2] = norm_z
    trans[3, 3] = 1

    mesh.apply_transform(trans)
    # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), norm_x))
    if not np.allclose(center, [0.0, 0.0, 0.0]):
        mesh.vertices += np.array(center)


def generate_cylinder_mesh(edges: np.array, n: int, r: float):
    vertices = []
    faces = []

    ang = 2 * np.pi / n
    offset = 0
    for edge in edges:
        vs = []
        fs = []
        center = (0.5 * (edge[0][0] + edge[1][0]), 0.5 * (edge[0][1] + edge[1][1]),
                  0.5 * (edge[0][2] + edge[1][2]))
        height = np.sqrt(np.power(edge[0][0] - edge[1][0], 2) + np.power(edge[0][1] - edge[1][1], 2)
                         + np.power(edge[0][2] - edge[1][2], 2))
        direction = (edge[0][0] - edge[1][0], edge[0][1] - edge[1][1], edge[0][2] - edge[1][2])
        x = 0.5 * height
        for i in range(n):
            z = r * np.cos(i * ang)
            y = r * np.sin(i * ang)
            vs.append(np.array([x, y, z]))

        x = -0.5 * height
        for i in range(n):
            z = r * np.cos(i * ang)
            y = r * np.sin(i * ang)
            vs.append(np.array([x, y, z]))

        for i in range(n):
            fs.append(np.array([
                i, (i + 1) % n, i + n
            ]))
            fs.append(np.array([
                i + n, (i + 1) % n, (i + n + 1) % n + n
            ]))

        mesh = trimesh.Trimesh(vs, fs)
        transform(mesh, center, direction)
        vs = list(np.asarray(mesh.vertices))
        vertices.extend(vs)
        for f in fs:
            f += offset
        faces.extend(fs)
        offset += len(vs)

    return vertices, faces


def generate_bounding_box_vertices_mesh(vertices: np.array, faces: np.array, n: int, r: float):
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    bounds = mesh.bounds
    points = [
        bounds[0],
        np.array([
            bounds[1][0], bounds[0][1], bounds[0][2]
        ]),
        np.array([
            bounds[1][0], bounds[0][1], bounds[1][2]
        ]),
        np.array([
            bounds[0][0], bounds[0][1], bounds[1][2]
        ]),
        np.array([
            bounds[0][0], bounds[1][1], bounds[0][2]
        ]),
        np.array([
            bounds[1][0], bounds[1][1], bounds[0][2]
        ]),
        bounds[1],
        np.array([
            bounds[0][0], bounds[1][1], bounds[1][2]
        ]),
        0.5 * (bounds[0] + bounds[1])
    ]

    return generate_cube_mesh(points, n, r)


def generate_bounding_box_edges_mesh(vertices: np.array, faces: np.array, n: int, r: float):
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    bounds = mesh.bounds
    joints = [
        bounds[0],
        np.array([
            bounds[1][0], bounds[0][1], bounds[0][2]
        ]),
        np.array([
            bounds[1][0], bounds[0][1], bounds[1][2]
        ]),
        np.array([
            bounds[0][0], bounds[0][1], bounds[1][2]
        ]),
        np.array([
            bounds[0][0], bounds[1][1], bounds[0][2]
        ]),
        np.array([
            bounds[1][0], bounds[1][1], bounds[0][2]
        ]),
        bounds[1],
        np.array([
            bounds[0][0], bounds[1][1], bounds[1][2]
        ])
    ]

    edges = [
        [joints[0], joints[1]],
        [joints[0], joints[3]],
        [joints[0], joints[4]],
        [joints[1], joints[2]],
        [joints[1], joints[5]],
        [joints[2], joints[6]],
        [joints[2], joints[3]],
        [joints[3], joints[7]],
        [joints[4], joints[7]],
        [joints[4], joints[5]],
        [joints[5], joints[6]],
        [joints[6], joints[7]],
        [joints[0], joints[6]],
        [joints[1], joints[7]],
        [joints[2], joints[4]],
        [joints[3], joints[5]]
    ]

    return generate_cylinder_mesh(edges, n, r)


class Renderer:

    def __init__(self, bg_color: np.array = np.array((1.0, 1.0, 1.0, 0.0))):
        self.img_visual_size = 900
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.3, 0.3, 0.3))

    def add_mesh(self, vertices, faces, rotation, color, alpha_mode: str = "BLEND"):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.5,
            alphaMode=alpha_mode,
            baseColorFactor=color
        )

        mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rotation[0]), [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rotation[1]), [0, 1, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rotation[2]), [0, 0, 1]))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        self.scene.add(mesh, 'mesh')

    def render(self, visible_list: List[Visible], image_size: np.array, camera_translation: np.array,
               camera_rotation: np.array = np.array((1, 0, 0)),
               rotation: np.array = np.array((0, 0, 0)),
               focal_length: np.array = np.array((5000., 5000.))):
        self.scene.clear()

        for visible in visible_list:
            if visible.visible_type == VisibleTypes.MESH:
                self.add_mesh(visible.vertices, visible.faces, rotation, visible.color)
            elif visible.visible_type == VisibleTypes.POINTS:
                p_vertices, p_faces = generate_cube_mesh(visible.points, 16, 0.008)
                self.add_mesh(p_vertices, p_faces, rotation, visible.color, "OPAQUE")
            elif visible.visible_type == VisibleTypes.LINES:
                l_vertices, l_faces = generate_cylinder_mesh(visible.lines, 20, 0.003)
                self.add_mesh(l_vertices, l_faces, rotation, visible.color, "OPAQUE")

        up_scale = float(max(image_size[0], image_size[1])) / 256
        cam_translation = camera_translation.copy()
        cam_translation[2] /= up_scale
        cam_translation[0] *= -1.
        camera_pose = gen_transform_matrix(cam_translation, camera_rotation)
        camera_center = [image_size[1] / 2., image_size[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                           cx=camera_center[0], cy=camera_center[1], zfar=1000)

        self.scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            self.scene.add_node(node)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=image_size[1],
            viewport_height=image_size[0],
            point_size=1.0
        )
        color, rend_depth = renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        color = color.astype(np.float32) / 255.0
        output_img = color

        return output_img

    @staticmethod
    def add_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
