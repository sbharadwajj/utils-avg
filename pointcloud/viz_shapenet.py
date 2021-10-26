import open3d as o3d 
import numpy as np
import os
import sys


def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

def points_pcd(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

path = sys.argv[1]
save_path = path + 'shapenet-red-rot' 
if not os.path.exists(save_path):
    os.mkdir(save_path)
# import pdb; pdb.set_trace()
for path in [path]:
    batch = np.load(path)

    preds = batch['predictions']
    inp = batch['data']
    gt = batch['gt']

    cnt = 0
    for pred, ip, gt in zip(preds, inp, gt):
        i = cnt
        cnt+=1
        # import pdb;pdb.set_trace()
        if cnt in [10, 25, 119, 149, 209, 249, 309, 319, 399, 409, 419, 449, 499, 599, 609, 629, 719]:
            pcd_preds = points_pcd(pred)
            pcd_inp = points_pcd(ip.transpose(1,0))
            pcd_gt = points_pcd(gt)
            
            pcd_gt.paint_uniform_color(np.array([1, 0, 0]))
            pcd_preds.paint_uniform_color(np.array([1, 0, 0]))
            pcd_inp.paint_uniform_color(np.array([1, 0, 0]))

            # R = np.asarray([(0,1,0),(-1,0,0),(0,0,1)]).astype(np.float64)
            # R_2 = np.asarray([(0,0,1),(0,1,0),(-1,0,0)]).astype(np.float64)
            # center_gt = pcd_gt.get_center().astype(np.float64)
            # pcd_gt.rotate(R, center_gt)
            # pcd_gt.rotate(R_2, center_gt)

            # pcd_preds.rotate(R, center_gt)
            # pcd_preds.rotate(R_2, center_gt)

            # pcd_inp.rotate(R, center_gt)
            # pcd_inp.rotate(R_2, center_gt)

            # o3d.visualization.draw_geometries_with_animation_callback([pcd_gt],rotate_view)
            # o3d.visualization.draw_geometries_with_animation_callback([pcd_inp],rotate_view)
            # o3d.visualization.draw_geometries_with_animation_callback([pcd_preds],rotate_view)
            



            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd_inp)
            # vis.add_geometry(pcd_gt)
            vis.update_geometry(pcd_inp)
            # vis.update_geometry(pcd_gt)
            ctr = vis.get_view_control()
            # ctr.scale(25)
            ctr.set_zoom(1.5)
            vis.run()
            vis.capture_screen_image(os.path.join(save_path,str(i) + "partial.png"))
            vis.destroy_window()

            del vis
            del ctr

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            # vis.add_geometry(pcd_inp)
            vis.add_geometry(pcd_preds)
            # vis.update_geometry(pcd_inp)
            ctr = vis.get_view_control()
            # ctr.scale(25)
            ctr.set_zoom(1.5)
            vis.run()
            vis.capture_screen_image(os.path.join(save_path,str(i) + "preds.png"))
            vis.destroy_window()

            del vis
            del ctr

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd_gt)
            # vis.add_geometry(pcd_preds)
            vis.update_geometry(pcd_gt)
            # vis.update_geometry(pcd_preds)
            ctr = vis.get_view_control()
            # ctr.scale(25)
            ctr.set_zoom(1.5)
            vis.run()
            vis.capture_screen_image(os.path.join(save_path,str(i) + "gt.png"))
            vis.destroy_window()

            del vis
            del ctr
        # o3d.visualization.draw_geometries([pcd_preds])
        # o3d.visualization.draw_geometries([pcd_gt])
        # o3d.visualization.draw_geometries([pcd_inp])
        
        