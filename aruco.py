import os
import shutil

from argparse import ArgumentParser
from aruco_estimator.sfm.colmap import COLMAPProject
from aruco_estimator.utils import *
from aruco_estimator.visualization import VisualizationModel
import cv2

# https://github.com/meyerls/aruco-estimator
parser = ArgumentParser("Automatic Registration Estimation Based on ArUco Markers")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--target_id", "-t", default=0, type=int)
parser.add_argument("--aruco_size", "-a", default=0.1, type=float)
parser.add_argument("--show", "-v", action="store_true")
args = parser.parse_args()
source = args.source_path
target_id = args.target_id
aruco_size = args.aruco_size
output_path = os.path.normpath(source) + "_norm/"


project = COLMAPProject(source)

print("Detecting ArUco markers...")
aruco_results = project.detect_markers(dict_type=cv2.aruco.DICT_4X4_50)
target_corners_3d = aruco_results[target_id]
transform = get_transformation_between_clouds(target_corners_3d, get_corners_at_origin(side_length=aruco_size))


print("Normalizing poses and 3D points...")
project.transform(transform)
project.save(os.path.join(output_path, "sparse/0"))

shutil.copytree(os.path.join(source, "input"), os.path.join(output_path, "input"), dirs_exist_ok=True)
shutil.copytree(os.path.join(source, "images"), os.path.join(output_path, "images"), dirs_exist_ok=True)
if os.path.exists(os.path.join(source, "masks")):
    shutil.copytree(os.path.join(source, "masks"), os.path.join(output_path, "masks"), dirs_exist_ok=True)


if args.show:
    model = VisualizationModel()
    model.create_window()

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    aligned_projects = [project]

    for i, proj in enumerate(aligned_projects):
        color = colors[i % len(colors)]
        model.add_project(
            proj,
            points_config={"color": color},
            cameras_config={"scale": 0.25, "color": color},
            markers_config={
                "show_detection_lines": True,
                "detection_line_color": color,
                "corner_size": 0.05,
            },
        )

    model.add_coordinate_frame(size=aruco_size)
    model.show()