import open3d as o3d 
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


def points_pcd(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd


if __name__ == "__main__":
    input_folder = sys.argv[1]
    assert(os.path.exists(input_folder))
    for f in os.listdir(input_folder):
        input_path = os.path.join(input_folder, f)
        input_pcd = points_pcd(np.load(input_path))

        # R = np.asarray([(-1,0,0),(0,1,0),(0,0,1)]).astype(np.float64)
        # R_2 = np.asarray([(0,0,1),(0,1,0),(-1,0,0)]).astype(np.float64)
        # center_gt = input_pcd.get_center().astype(np.float64)
        # input_pcd.rotate(R, center_gt)

        o3d.visualization.draw_geometries([input_pcd])