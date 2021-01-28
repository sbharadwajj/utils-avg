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


def points_pcd(point_set, center):
    pcd = o3d.geometry.PointCloud()
    point_set = point_set - center.transpose()
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

if __name__ == "__main__":
    input_folder = sys.argv[1]
    gt_folder = sys.argv[2]
    assert(os.path.exists(input_folder))
    save_folder = sys.argv[3]
    poses = (np.loadtxt(sys.argv[4])).astype(np.float64)

    if sys.argv[5] == 'save':
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    for f in ["106.dat", "22.dat", "269.dat", "370.dat", "429.dat", "555.dat", "670.dat", "750.dat"]:#os.listdir(input_folder):
        input_path = os.path.join(input_folder, f)
        gt_path = os.path.join(gt_folder, f)
        input_points = (np.loadtxt(input_path)).astype(np.float64)
        gt_points = (np.loadtxt(gt_path)).astype(np.float64)
        x = int(f.split('.')[0])
        if x in poses[:,0]:
            pose_x = poses[poses[:,0]==x] # check the index of the pose
            pose_matrix = pose_x[0,1:].reshape(3,4)
            translation_vec = pose_matrix[:,3:].astype(np.float64)
            input_pcd = points_pcd(input_points, translation_vec)
            gt_pcd = points_pcd(gt_points, translation_vec)
            input_pcd.paint_uniform_color(np.array([1, 0.706, 0]))
            gt_pcd.paint_uniform_color(np.array([1, 0, 0]))

            if sys.argv[5] == 'save':
                print("saving image:"+f)
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(input_pcd)
                vis.add_geometry(gt_pcd)
                vis.update_geometry(input_pcd)
                vis.update_geometry(gt_pcd)
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(os.path.join(save_folder,str(x) + ".png"))
                vis.destroy_window()
            else:
                o3d.visualization.draw_geometries([input_pcd, gt_pcd])