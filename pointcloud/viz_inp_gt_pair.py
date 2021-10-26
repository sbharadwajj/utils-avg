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
import transforms3d
import pandas as pd
import random
import math

def downsample(inputpcd, voxel_size):
    downpcd = inputpcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points).astype(np.float64), downpcd

def resample_pcd(gt_aug_pcd, n):
    """Drop or duplicate points so that gt_aug_pcd has exactly n points"""
    idx = np.random.permutation(gt_aug_pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(gt_aug_pcd.shape[0], size = n - gt_aug_pcd.shape[0])])
    return gt_aug_pcd[idx[:n]]

def points_pcd(point_set):
    gt_aug_pcd = o3d.geometry.PointCloud()
    # point_set = point_set - center.transpose()
    # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    # point_set = point_set / dist #scale
    gt_aug_pcd.points = o3d.utility.Vector3dVector(point_set) 
    return gt_aug_pcd

def get_thresh(points_z, percent, array_pcd, axis):
    thresh = np.quantile(points_z, percent)
    bottom = array_pcd[array_pcd[:, axis] < thresh]
    top = array_pcd[array_pcd[:, axis] > thresh]
    return bottom, top

def augment_cloud(Ps):
    """" Augmentation on XYZ and jittering of everything """
    "Augmented params:"
    pc_augm_scale=0
    pc_augm_rot=1
    pc_augm_mirror_prob=0.5
    pc_augm_jitter=0

    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_scale > 1:
        s = random.uniform(1/pc_augm_scale, pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_rot:
        angle = random.uniform(0, 2*math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # z=upright assumption
    if pc_augm_mirror_prob > 0: # mirroring x&y, not z
        if random.random() < pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
    result = []
    for P in Ps:
        P[:,:3] = np.dot(P[:,:3], M.T)

        if pc_augm_jitter:
            sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
        result.append(P)
    return result

if __name__ == "__main__":
    input_folder = sys.argv[1]
    gt_folder = sys.argv[2]
    assert(os.path.exists(input_folder))
    save_folder = sys.argv[3]
    poses = (np.loadtxt(sys.argv[4])).astype(np.float64)

    if sys.argv[5] == 'save':
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    for f in os.listdir(input_folder):#["106.dat", "22.dat", "269.dat", "370.dat", "429.dat", "555.dat", "670.dat", "750.dat"]
        input_path = os.path.join(input_folder, f)
        gt_path = os.path.join(gt_folder, f)
        input_points = (np.load(input_path)).astype(np.float64)
        gt_points = (np.load(gt_path)).astype(np.float64)

        axis = 2
        # check
        # input_points, gt_points = get_thresh(gt_points[:, axis], 0.7, gt_points, axis)
        # y_axis = np.full((gt_points.shape[0], 1), gt_points[:,axis].min())
        # gt_points[:, axis] = np.squeeze(y_axis)

        input_pcd = points_pcd(input_points)
        gt_pcd = points_pcd(gt_points)

        # input_pcd.paint_uniform_color(np.array([1, 0.706, 0]))
        # gt_pcd.paint_uniform_color(np.array([0, 1, 0]))
        # x = int(f.split('.')[0])
        # if x in poses[:,0]:
        #     pose_x = poses[poses[:,0]==x] # check the index of the pose
        #     pose_matrix = pose_x[0,1:].reshape(3,4)
        #     translation_vec = pose_matrix[:,3:].astype(np.float64)
        #     input_pcd = points_pcd(input_points, translation_vec)
        #     gt_pcd = points_pcd(gt_points, translation_vec)


        # input_pcd.paint_uniform_color(np.array([1, 0.706, 0]))
        # gt_pcd.paint_uniform_color(np.array([1, 0, 0]))
            
        #     print(gt_pcd)
        #     down_gt = resample_pcd(gt_points, 8192)
        #     pcd_down = o3d.geometry.PointCloud()
        #     pcd_down.points = o3d.utility.Vector3dVector(down_gt) 
        #     print(pcd_down)

        gt_aug, inp_aug = augment_cloud([gt_points, input_points])
        gt_aug_pcd = o3d.geometry.PointCloud()
        gt_aug_pcd.points = o3d.utility.Vector3dVector(gt_aug) 
        inp_aug_pcd = o3d.geometry.PointCloud()
        inp_aug_pcd.points = o3d.utility.Vector3dVector(inp_aug) 
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
            o3d.visualization.draw_geometries([gt_pcd])
            o3d.visualization.draw_geometries([input_pcd])
