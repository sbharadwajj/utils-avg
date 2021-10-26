import open3d as o3d 
import numpy as np
import os
import sys
import torch
import time
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

def rotate_view_gt(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    path = "final_baseline_viz/GIF/p++/gt/"
    if not os.path.exists(path):
        os.mkdir(path)
    vis.capture_screen_image(path+str(time.time())+"gt.png")
    return False

def rotate_view_pred(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    path = "final_baseline_viz/GIF/p++/pred/"
    if not os.path.exists(path):
        os.mkdir(path)
    vis.capture_screen_image(path+str(time.time())+"pred.png")
    return False

def points_pcd(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

def points_pcd_norm(point_set,center):
    pcd = o3d.geometry.PointCloud()
    point_set = point_set - center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

def points_pcd_color(point_set, color):
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.points = o3d.utility.Vector3dVector(point_set.transpose()) 
    return pcd

path = sys.argv[1]
save_path = path + '_viz'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# import pdb; pdb.set_trace()
for npz in os.listdir(path):
    batch = np.load(os.path.join(path,npz))
    
    for i in range(1):#range(batch['pred'].shape[0]):##
        pred_ = batch['pred'][i]

        sm = torch.nn.Softmax(dim=0)
        softmax_pred = sm(torch.from_numpy(pred_)).numpy()

        pred = np.argmax(softmax_pred, axis=0)
        
        feat_color = pred[pred != 19]
        mask_arr = pred
        mask_arr[mask_arr != 19] = 1.0
        mask_arr[mask_arr == 19] = 0.0
        np_points = np.vstack(np.where(mask_arr==1))

        colors_mapped = []
        for color in feat_color:
            colors_mapped.append(labels_20_classes[color])

        colors_feat = (np.asarray(colors_mapped) / 255).astype(np.float32)
        pcd_pred = points_pcd_color(np_points, colors_feat)
        vox_pred = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_pred,voxel_size=0.5)

        gt = batch['gt'][i]
        
        feat_color_gt = gt[gt != 19]
        mask_arr_gt = gt
        mask_arr_gt[mask_arr_gt != 19] = 1.0
        mask_arr_gt[mask_arr_gt == 19] = 0.0
        np_points_gt = np.vstack(np.where(mask_arr_gt==1))

        colors_mapped_gt = []
        for color in feat_color_gt:
            colors_mapped_gt.append(labels_20_classes[color])

        colors_feat_gt = (np.asarray(colors_mapped_gt) / 255).astype(np.float32)
        pcd_gt = points_pcd_color(np_points_gt, colors_feat_gt)

        vox_gt = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_gt,voxel_size=0.5)

        '''
        THIS CONVERSION IS CORRECT, THE NUMBER OF OCCUPIED VOXELS IS THE SAME,
        vox size only controls visualization and nothing else
        '''

        partial_points = batch['inp'][i]
        pcd_inp = points_pcd(partial_points)
        # o3d.visualization.draw_geometries([pcd_pred])
        # # o3d.visualization.draw_geometries([pcd_partial, vox_gt])
        # o3d.visualization.draw_geometries([pcd_gt])

        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_gt)
        ctr = vis.get_view_control()
        # ctr.scale(25)
        ctr.set_zoom(1.5)
        vis.register_animation_callback(rotate_view_gt)
        vis.run()
        # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "gt.png"))
        vis.destroy_window()

        del vis
        del ctr

        vis = o3d.visualization.Visualizer()
        vis.create_window()        
        vis.add_geometry(pcd_pred)
       
        ctr = vis.get_view_control()
        # ctr.scale(25)
        ctr.set_zoom(1.5)
        vis.register_animation_callback(rotate_view_pred)
        vis.run()
        vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "preds.png"))
        vis.destroy_window()

        del vis
        del ctr

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_inp)

        # ctr = vis.get_view_control()
        # # ctr.scale(25)
        # ctr.set_zoom(1.5)
        # vis.run()
        # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "partial.png"))
        # vis.destroy_window()

        # del vis
        # del ctr