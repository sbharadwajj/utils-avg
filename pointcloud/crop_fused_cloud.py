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
import matplotlib.pyplot as plt
import os
import struct
import sys

from read_semantic_labels import loadWindow

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
    return pcd

if __name__ == "__main__":
    input_folder = sys.argv[1]
    assert(os.path.exists(input_folder))
    save_folder = sys.argv[2]
    radius = int(sys.argv[4])

    for f in os.listdir(input_folder):
        input_path = os.path.join(input_folder, f)
        input_pcd = o3d.io.read_point_cloud(input_path)
        poses = (np.loadtxt(sys.argv[3])).astype(np.float64)
        if sys.argv[5] == 'semantic':
            input_pcd = loadWindow(input_path, input_pcd)
        name_split = f.split('_')
        f_start, f_end = name_split[0], name_split[1].split('.')[0]
        X_files = get_X_names(int(f_start), int(f_end))

        for x in X_files:
            if x in poses[:,0]:
                pose_x = poses[poses[:,0]==x] # check the index of the pose
                pose_matrix = pose_x[0,1:].reshape(3,4)
                translation_vec = pose_matrix[:,3:].astype(np.float64)
                cropped_pcd = crop_pointcloud(input_pcd, radius, translation_vec)
                o3d.visualization.draw_geometries([cropped_pcd])
                