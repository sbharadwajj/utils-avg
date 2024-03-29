'''
voxel_size:
1. 0.5 reduces the point cloud by 20 times (i.e 100000 --> 5000)
2. 0.05 reduces the point cloud by 12 times (i.e 100000 --> 80000)

run:
`python voxelize_pointcloud.py voxel_size input_folder save_folder start_name_file end_name_file ply`

example:
`python voxelize_pointcloud.py 0.5 input_folder downsampled_train start_name_file 800 1200 ply`

'''

import open3d as o3d 
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import sys


# VOXELIZE HERE
def voxelize(inputpcd, voxel_size):
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(inputpcd,voxel_size)
    import pdb;pdb.set_trace()
    return voxel

def file_to_pcd(path):
    points = (np.loadtxt(path)).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  
    return pcd

if __name__ == "__main__":
    input_folder = sys.argv[2]
    assert(os.path.exists(input_folder))
    save_folder = sys.argv[3]
    for f in os.listdir(input_folder):
        # if int(f.split('.')[0]) > int(sys.argv[4]) and int(f.split('.')[0]) < int(sys.argv[5]):
        input_path = os.path.join(input_folder, f)
        voxel_size = float(sys.argv[1])
        if sys.argv[4] == 'ply':
            input_pcd = o3d.io.read_point_cloud(input_path)
        else:
            input_pcd = file_to_pcd(input_path)
        N = np.asarray(input_pcd.points).shape[0]
        voxel_grid = voxelize(input_pcd, voxel_size)
        print(voxel_grid.get_oriented_bounding_box)
        o3d.visualization.draw_geometries([voxel_grid])

        # save_path = save_folder + f.split(".")[0] + ".binvox"
        # print(save_path)
        # import pdb;pdb.set_trace()
        # o3d.io.write_voxel_grid(save_path, voxel_grid)