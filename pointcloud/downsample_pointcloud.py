'''
voxel_size:
1. 0.5 reduces the point cloud by 20 times (i.e 100000 --> 5000)
2. 0.05 reduces the point cloud by 12 times (i.e 100000 --> 80000)

run:
`python downsample_pointcloud.py voxel_size input_folder save_folder start_name_file end_name_file ply`

example:
`python downsample_pointcloud.py 0.5 input_folder downsampled_train start_name_file 800 1200 ply`

'''

import open3d as o3d 
import numpy as np
import os
import struct
import sys

def downsample(inputpcd, voxel_size):
    downpcd = inputpcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points).astype(np.float64)

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
        if int(f.split('.')[0]) > int(sys.argv[4]) and int(f.split('.')[0]) < int(sys.argv[5]):
            input_path = os.path.join(input_folder, f)
            voxel_size = float(sys.argv[1])
            if sys.argv[6] == 'ply':
                input_pcd = o3d.io.read_point_cloud(input_path)
            else:
                input_pcd = file_to_pcd(input_path)
            downpcd = downsample(input_pcd, voxel_size)
            save_path = os.path.join(save_folder, str(int(f.split('.')[0])) + '.dat')
            print("downsampled point cloud saved as array at:" + save_path)
            np.savetxt(save_path, downpcd)
        