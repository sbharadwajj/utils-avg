import open3d as o3d 
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import imageio

import time

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    # path = "final_baseline_viz/GIF/p++/gt/"
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # vis.capture_screen_image(path+str(time.time())+"gt.png")
    return False

def rotate_view_partial(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    path = "final_baseline_viz/GIF/p++/partial/"
    if not os.path.exists(path):
        os.mkdir(path)
    vis.capture_screen_image(path+str(time.time())+"partial.png")
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

def get_thresh(points_z, percent, array_pcd, axis):
    thresh = np.quantile(points_z, percent)
    print(thresh)
    bottom = array_pcd[array_pcd[:, axis] < thresh]
    top = array_pcd[array_pcd[:, axis] > thresh]
    print(bottom.shape)
    print(top.shape)
    return bottom, top

def points_pcd_color(pcd):
    norm = plt.Normalize(vmin=-0.1, vmax=0.2)
    pcd_points = np.asarray(pcd.points)
    colors = plt.cm.jet(norm(pcd_points[:, 2]))
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    #return pcd

feat = False
path = sys.argv[1]
save_path = path + 'TESTTTT' 
if not os.path.exists(save_path):
    os.mkdir(save_path)
# import pdb; pdb.set_trace()
for path in [path]:
    batch = np.load(path)

    preds = batch['predictions']
    inp = batch['data']
    gt = batch['gt']

    if feat:
        feats = batch['feat']

    cnt = 0
    for pred, ip, gt in zip(preds, inp, gt): 
        if cnt in [0,8]:
            axis = 2
            # check
            # ip, gt = get_thresh(gt[:, axis], 0.6, gt, axis)
                
            pcd_preds = points_pcd(pred)
            pcd_inp = points_pcd(ip.transpose(1,0))
            # pcd_inp = points_pcd(ip)
            pcd_gt = points_pcd(gt)
            # import pdb;pdb.set_trace()
            
            if feat:
                globalfeat = points_pcd(feats[cnt])

            # pcd_gt.paint_uniform_color(np.array([1, 0.706, 0]))
            # pcd_preds.paint_uniform_color(np.array([0, 1, 0]))
            # pcd_inp.paint_uniform_color(np.array([1, 0, 0]))
            
            points_pcd_color(pcd_gt)
            points_pcd_color(pcd_inp)
            points_pcd_color(pcd_preds)
            # o3d.visualization.draw_geometries([pcd_inp, pcd_gt])
            # o3d.visualization.draw_geometries([pcd_preds, pcd_gt])

            if feat:
                o3d.visualization.draw_geometries([globalfeat])
            
            # o3d.visualization.draw_geometries_with_animation_callback([pcd_inp],rotate_view)
            # o3d.visualization.draw_geometries_with_animation_callback([pcd_preds],rotate_view)
            o3d.visualization.draw_geometries_with_animation_callback([pcd_gt],rotate_view)

            # for i in range(10):
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            # vis.add_geometry(pcd_gt)
            # ctr = vis.get_view_control()
            # # ctr.scale(25)
            # # ctr.rotate(10.0, 0.0)
            # ctr.set_zoom(1.5)
            # vis.register_animation_callback(rotate_view_gt)
            # vis.run()
            # vis.destroy_window()

            # del vis
            # del ctr

            vis = o3d.visualization.Visualizer()
            vis.create_window()        
            vis.add_geometry(pcd_preds)
        
            ctr = vis.get_view_control()
            # ctr.scale(25)
            ctr.set_zoom(1.5)
            vis.register_animation_callback(rotate_view_pred)
            vis.run()
            # vis.capture_screen_image(os.path.join(save_path,str(cnt) + "preds.png"))
            vis.destroy_window()

            del vis
            del ctr

            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            # vis.add_geometry(pcd_inp)

            # ctr = vis.get_view_control()
            # # ctr.scale(25)
            # ctr.set_zoom(1.5)
            # vis.register_animation_callback(rotate_view_partial)
            # vis.run()
            # # vis.capture_screen_image(os.path.join(save_path,str(cnt) + "partial.png"))
            # vis.destroy_window()

            # del vis
            # del ctr


        cnt+=1

        # if cnt > 15:
        #     break
    # o3d.visualization.draw_geometries([pcd_preds])
    # o3d.visualization.draw_geometries([pcd_gt])
    # o3d.visualization.draw_geometries([pcd_inp])
        
        