'''
run:
`python crop_fused_cloud.py input_folder save_folder pose_file radius_cropping semantic/none voxel/none save/viz`
example:
`python crop_fused_cloud.py data_3d_semantics/2013_05_28_drive_0000_sync/static/ tmp/ data_poses/2013_05_28_drive_0000_sync/poses.txt 70 semantic voxel save`
my code
`python crop_fused_cloud.py ../../voxel_data/fused_cloud/ ../../voxel_data/viz/ ../../data_poses/2013_05_28_drive_0000_sync/poses.txt 70 semantic voxel save`
'''

import open3d as o3d 
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import sys
import pandas as pd

from pyntcloud import PyntCloud

import binvox_rw

from read_semantic_labels import loadWindow
from voxelize_pointcloud import voxelize

# def voxelize(points, x, y, z):
#     cloud = PyntCloud(points, sep=" ", header=0, names=["x", "y", "z"])
#     voxelgrid_id = cloud.add_structure("voxelgrid", n_x=x, n_y=y, n_z=z)
#     voxelgrid = cloud.structures[voxelgrid_id]
#     # voxelgrid.plot(d=3, mode="density", cmap="hsv")

#     x_cords = voxelgrid.voxel_x
#     y_cords = voxelgrid.voxel_y
#     z_cords = voxelgrid.voxel_z

#     voxel = np.zeros((x, y, z)).astype(np.bool)

#     for x, y, z in zip(x_cords, y_cords, z_cords):
#         voxel[x][y][z] = True

#     with open("00000.binvox", 'wb') as f:
#         v = binvox_rw.Voxels(voxel, (x, y, z), (0, 0, 0), 1, 'xyz')
#         v.write(f)
#     return voxel


def get_X_names(start, end):
    X_list = []
    for n in range((end-start)+1):
        X_list.append(start+n)
    assert(X_list[0] == start)
    assert(X_list[len(X_list)-1] == end)
    return X_list

def file_to_pcd(path):
    points = (np.loadtxt(path)).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  
    return pcd

def crop_pointcloud(cloud, r, ref):
    '''
    crop pcd along with semantic labels
    '''
    points = np.asarray(cloud.points).astype(np.float64)
    colors = np.asarray(cloud.colors)
    dist = np.linalg.norm(np.transpose(ref) - points, axis=1)
    bounding_box = points[(dist < r)]
    colors_box = colors[(dist < r)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bounding_box)
    pcd.colors = o3d.utility.Vector3dVector(colors_box)
    return pcd, bounding_box

if __name__ == "__main__":
    input_folder = sys.argv[1]
    assert(os.path.exists(input_folder))
    save_folder = sys.argv[2]
    radius = int(sys.argv[4])
    poses = (np.loadtxt(sys.argv[3])).astype(np.float64)

    if sys.argv[7] == 'save':
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    for f in os.listdir(input_folder):
        input_path = os.path.join(input_folder, f)
        input_pcd = o3d.io.read_point_cloud(input_path)
        if sys.argv[5] == 'semantic':
            input_pcd = loadWindow(input_path, input_pcd)
        name_split = f.split('_')
        f_start, f_end = name_split[0], name_split[1].split('.')[0]
        X_files = get_X_names(int(f_start), int(f_end))

        for x in X_files:
            if x in poses[:,0] and x in [10, 100, 150, 200, 250, 400, 600, 800]:
                pose_x = poses[poses[:,0]==x] # check the index of the pose
                pose_matrix = pose_x[0,1:].reshape(3,4)
                translation_vec = pose_matrix[:,3:].astype(np.float64)
                cropped_pcd, cropped_points = crop_pointcloud(input_pcd, radius, translation_vec) # cropped pcd
                o3d.io.write_point_cloud(os.path.join(save_folder,str(x)+"_cropped.ply"), cropped_pcd)
                if sys.argv[6] == 'voxel':
                    # Bug 
                    voxel_grid = voxelize(cropped_pcd, 0.4)

                if sys.argv[7] == 'save':
                    vis = o3d.visualization.Visualizer()
                    vis.create_window()
                    vis.add_geometry(voxel_grid)
                    vis.update_geometry(voxel_grid)
                    vis.poll_events()
                    vis.update_renderer()
                    vis.capture_screen_image(os.path.join(save_folder,str(x) + "_voxel.png"))
                    vis.destroy_window()
                # else:
                #     o3d.visualization.draw_geometries([voxel_grid])
