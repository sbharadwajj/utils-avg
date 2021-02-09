'''
run:
`python utils-avg/pointcloud/create_train_pair.py 2013_05_28_drive_train_only.txt path_fused_cloud save_folder radius pose_folder save`

example:
`python utils-avg/pointcloud/create_train_pair.py 2013_05_28_drive_train_only.txt data_3d_semantics/ baseline_data/train_fused/ 70 data_poses/ save`

my code
`python utils-avg/pointcloud/create_train_pair.py 2013_05_28_drive_train_only.txt data_3d_semantics/ baseline_data/train_fused/ 70 data_poses/ save`
'''

import open3d as o3d 
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import sys
import os.path as osp
import h5py

voxel_size = 0.5
n = 5000

def get_X_names(start, end):
    X_list = []
    for n in range((end-start)+1):
        X_list.append(start+n)
    assert(X_list[0] == start)
    assert(X_list[len(X_list)-1] == end)
    return X_list

def points_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  
    return pcd

def crop_pointcloud(cloud, r, ref):
    '''
    crop pcd along with semantic labels
    '''
    points = np.asarray(cloud.points).astype(np.float64)
    dist = np.linalg.norm(np.transpose(ref) - points, axis=1)
    bounding_box = points[(dist < r)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bounding_box)
    return pcd

def downsample(inputpcd, voxel_size, n):
    downpcd = inputpcd.voxel_down_sample(voxel_size=voxel_size)
    pcd = np.asarray(downpcd.points).astype(np.float64)
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

if __name__ == "__main__":
    train_file = sys.argv[1]
    train_seq = (np.loadtxt(train_file)).astype(np.int16)
    
    input_folder = sys.argv[2]
    assert(osp.exists(input_folder))
    save_folder = sys.argv[3]
    radius = int(sys.argv[4])
    pose_folder = sys.argv[5]

    if sys.argv[6] == 'save':
        if not osp.exists(save_folder):
            os.mkdir(save_folder)

    num = 0
    for idx in np.unique(train_seq[:,0]): # loop for different drive folders
        drive_name = "2013_05_28_drive_%04d_sync" %idx
        fused_cloud_folder = osp.join(input_folder, drive_name,"static")
        print("Opening the folder:" + fused_cloud_folder)
        pose_path = osp.join(pose_folder, drive_name, "poses.txt")
        assert(osp.exists(pose_path))
        poses = np.loadtxt(pose_path).astype(np.float16)
        assert(osp.exists(fused_cloud_folder))
        seq_in_drive = train_seq[train_seq[:,0] == idx]
        seq_path = [osp.join(fused_cloud_folder,"%06d_%06d.ply" %(arr[1], arr[2])) \
                    for arr in seq_in_drive]
        
        seq_list = []
        for seq, idx in zip(seq_path, range(len(seq_path))):
            assert(osp.exists(seq))
            input_pcd = o3d.io.read_point_cloud(seq) # read fused cloud
            X_files = get_X_names(seq_in_drive[idx][1], seq_in_drive[idx][2]) # get start & end frame files

            for x in X_files:
                if x in poses[:,0] and not x%25:
                    pose_x = poses[poses[:,0]==x] # check the index of the pose
                    pose_matrix = pose_x[0,1:].reshape(3,4)
                    translation_vec = pose_matrix[:,3:].astype(np.float64)
                    cropped_pcd = crop_pointcloud(input_pcd, radius, translation_vec) # cropped pcd
                    downpcd = downsample(cropped_pcd, voxel_size, n)
                    seq_list.append(downpcd)
            print("Number of files %04d"%len(seq_list))
            num+=len(seq_list)
            if sys.argv[6] == 'save':
                save_path = osp.join(save_folder,seq.split("/")[-1].split(".")[0])
                print("saved at:"+ save_path)
                # np.save(save_path, np.asarray(seq_list)) # float 16 is 300kb
                with h5py.File(save_path+'.h5', 'w') as hf:
                    dset = hf.create_dataset(drive_name, data=np.asarray(seq_list))
            else:
                vizpcd = points_to_pcd(downpcd)
                o3d.visualization.draw_geometries([vizpcd])
        
    print("Total number of training files: %04d"%num)
