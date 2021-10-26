'''
Note: Functions are taken from this repository: https://github.com/autonomousvision/kitti360Scripts

This function takes in a fused pcd file from kitti360 and adds semantic labels to it.

run:
`python read_semantic_labels.py input_folder save_folder`

example:
`python read_semantic_labels.py input_folder downsampled_train`
'''

import open3d as o3d 
from pyntcloud import PyntCloud
import numpy as np
import os
import sys
import struct
import matplotlib
from labels import name2label, id2label, kittiId2label


cmap = matplotlib.cm.get_cmap('Set1')
cmap_length = 9

def global2local(globalId):
    semanticId = globalId // 1000
    instanceId = globalId % 1000
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int), instanceId.astype(np.int)
    else:
        return int(semanticId), int(instanceId)

def getColor(idx):
    if idx==0:
        return np.array([0,0,0])
    return np.asarray(cmap(idx % cmap_length)[:3])*255.

def assignColor(globalIds, gtType='semantic'):
    if not isinstance(globalIds, (np.ndarray, np.generic)):
        globalIds = np.array(globalIds)[None]
    color = np.zeros((globalIds.size, 3))
    for uid in np.unique(globalIds):
        semanticId, instanceId = global2local(uid)
        if gtType=='semantic':
            color[globalIds==uid] = id2label[semanticId].color
        elif instanceId>0:
            color[globalIds==uid] = getColor(instanceId)
        else:
            color[globalIds==uid] = (96,96,96) # stuff objects in instance mode
    color = color.astype(np.float)/255.0
    return color

def readBinaryPly(pcdFile, n_pts):
    fmt = '=fffBBBiiB'
    fmt_len = 24
    with open(pcdFile, 'rb') as f:
        plyData = f.readlines()

    headLine = plyData.index(b'end_header\n')+1
    plyData = plyData[headLine:]
    plyData = b"".join(plyData)

    n_pts_loaded = len(plyData)/fmt_len
    assert(n_pts_loaded==n_pts)
    n_pts_loaded = int(n_pts_loaded)

    data = []
    for i in range(n_pts_loaded):
        pts=struct.unpack(fmt, plyData[i*fmt_len:(i+1)*fmt_len])
        data.append(pts)
    data=np.asarray(data)

    return data
        
def loadWindow(pcdFile, pcd, colorType='semantic', isLabeled=True, isDynamic=False):
    n_pts = np.asarray(pcd.points).shape[0]
    data = readBinaryPly(pcdFile, n_pts)

    # assign color
    if colorType=='semantic' or colorType=='instance':
        globalIds = data[:,7]
        ptsColor = assignColor(globalIds, colorType)
        pcd.colors = o3d.utility.Vector3dVector(ptsColor)
    return pcd      

if __name__ == "__main__":
    input_folder = sys.argv[1]
    assert(os.path.exists(input_folder))
    save_folder = sys.argv[2]
    for f in os.listdir(input_folder):
        input_path = os.path.join(input_folder, f)
        input_pcd = o3d.io.read_point_cloud(input_path)

        # R = np.asarray([(-1,0,0),(0,1,0),(0,0,1)]).astype(np.float64)
        # R_2 = np.asarray([(0,0,1),(0,1,0),(-1,0,0)]).astype(np.float64)
        # center_gt = input_pcd.get_center().astype(np.float64)
        # input_pcd.rotate(R, center_gt)

        o3d.visualization.draw_geometries([input_pcd])
        pcd = loadWindow(input_path, input_pcd)

        # input_pcd.rotate(R, center_gt)
        o3d.visualization.draw_geometries([input_pcd])