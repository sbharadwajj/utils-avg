'''
to save files, threshold = 1200, number of files = 2000
python training_pair_translation.py ../../data_poses/2013_05_28_drive_0000_sync/poses.txt 2000 none 1200 ../../baseline_data/

to viz plot of x,y of poses
python training_pair_translation.py ../../data_poses/2013_05_28_drive_0000_sync/poses.txt 2000 plot savefolder
'''

import numpy as np
import sys
import matplotlib.pyplot as plt


def thresholdPoses(translation_matrix, threshold, pose_idx_matrix, train=True):
    crop_ = np.squeeze(translation_matrix[:n,:])
    if train:
        idx = np.squeeze(crop_[:,:1] < threshold)
    else:
        idx = np.squeeze(crop_[:,:1] > threshold) #test
    pose_id = pose_idx_matrix[idx]
    translation_vec = crop_[idx]
    return np.vstack((pose_id, translation_vec[:,0],translation_vec[:,1],translation_vec[:,2]))

if __name__ == "__main__":
    poses = (np.loadtxt(sys.argv[1])).astype(np.float64)
    n = int(sys.argv[2])
    threshold = int(sys.argv[4])
    matrix_poses = poses[:,1:].reshape(-1, 3,4)
    pose_idx_matrix = poses[:n,0]
    translation_matrix = matrix_poses[:,:,3:].astype(np.float64)
    x_axis = np.squeeze(translation_matrix[:n, :1, :]).tolist()
    y_axis = np.squeeze(translation_matrix[:n, 1:2, :]).tolist()
    
    if sys.argv[3] == 'plot':
        fig_name = sys.argv[5] + "/x-y" + sys.argv[2] + sys.argv[4] +"plot.jpg"
        fig = plt.figure()
        plt.plot(x_axis, y_axis, label="pose")
        plt.xlabel('x-axis of pose', fontsize=10)
        plt.ylabel('y-axis of pose', fontsize=10)
        fig.savefig(fig_name)
        # plt.show()
    else:
        savepath_train = sys.argv[5] + 'train_files_' + sys.argv[2] + sys.argv[4] + '.dat'
        savepath_test = sys.argv[5] + 'test_files_' + sys.argv[2] + sys.argv[4] + '.dat'
        train_matrix = thresholdPoses(translation_matrix, threshold, pose_idx_matrix, train=True)
        test_matrix = thresholdPoses(translation_matrix, threshold, pose_idx_matrix, train=False)
        np.save(savepath_train, train_matrix)
        np.save(savepath_test, test_matrix)
