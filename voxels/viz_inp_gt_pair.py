import open3d as o3d 
import numpy as np
import os
import sys
import torch

labels_20_classes = {19:(0, 0, 0),
                    20:(0,   64, 64),
                    0:(128, 64,128),
                    1:(244, 35,232),
                    3:(102,102,156),
                    7:(220,220,  0),
                    8:(107,142, 35),
                    9:(152,251,152),
                    10:( 70,130,180),
                    11:(220,20,60),
                    12:(255,0,0),
                    13:(0,0,142),
                    14:(0,0,70),
                    15:(0,60,100),
                    16:(0,80,100),
                    17:(0,0,230),
                    18:(119,11,32),
                    6:(250, 170, 30),
                    2:(70,70,70),
                    4:(190,153,153),
                    5:(153,153,153),}


def points_pcd(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

def points_pcd_color(point_set, color):
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.points = o3d.utility.Vector3dVector(point_set.transpose()) 
    return pcd

path = sys.argv[1]
path2 = sys.argv[2]
save_path = path + '_viz'
if not os.path.exists(save_path):
    os.mkdir(save_path)


for gt in os.listdir(path):
    pred = np.load(os.path.join(path, gt))

    feat_color = pred[pred != 19]
    mask_arr = pred
    mask_arr[mask_arr != 19] = 1.0
    mask_arr[mask_arr == 19] = 0.0
    np_points = np.vstack(np.where(mask_arr==1))

    # import pdb;pdb.set_trace()
    colors_mapped = []
    for color in feat_color:
        colors_mapped.append(labels_20_classes[color])

    colors_feat = (np.asarray(colors_mapped) / 255).astype(np.float32)
    pcd_gt = points_pcd_color(np_points, colors_feat)

    vox_gt = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_gt,voxel_size=0.5)

    '''
    THIS CONVERSION IS CORRECT, THE NUMBER OF OCCUPIED VOXELS IS THE SAME,
    vox size only controls visualization and nothing else
    '''

    partial_points = np.load(os.path.join(path2, gt))
    pcd_partial = points_pcd(partial_points)
    # o3d.visualization.draw_geometries([vox_gt])
    o3d.visualization.draw_geometries([vox_gt])
    o3d.visualization.draw_geometries([pcd_partial, vox_gt])
    o3d.visualization.draw_geometries([pcd_gt])
    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # vis.add_geometry(pcd_inp)
    # vis.add_geometry(pcd_gt)
    # # vis.update_geometry(pcd_inp)
    # vis.update_geometry(pcd_gt)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "gt.png"))
    # vis.destroy_window()

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # vis.add_geometry(pcd_inp)
    # vis.add_geometry(pcd_preds)
    # # vis.update_geometry(pcd_inp)
    # vis.update_geometry(pcd_preds)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "preds.png"))
    # vis.destroy_window()

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd_inp)
    # # vis.add_geometry(pcd_preds)
    # vis.update_geometry(pcd_inp)
    # # vis.update_geometry(pcd_preds)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "inp.png"))
    # vis.destroy_window()
    # # o3d.visualization.draw_geometries([pcd_preds.translate(center), pcd_gt.translate(center), pcd_inp])
    # # o3d.visualization.draw_geometries([pcd_gt])
    # # o3d.visualization.draw_geometries([pcd_inp])