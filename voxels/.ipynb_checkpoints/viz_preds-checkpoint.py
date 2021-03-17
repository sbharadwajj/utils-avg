
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D 
import sys
import numpy as np
# import streamlit as st

from simple_3dviz import Mesh
from simple_3dviz.window import show

if __name__ == "__main__":
    path = sys.argv[1]
    assert(os.path.exists(path))
    batch = np.load(path)
    for i in range(30):
        gt = batch['gt'][i]
        pred = batch['pred'][i]
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        # and plot everything
        # azim = st.sidebar.slider("azim", 0, 90, 30, 1)
        # elev = st.sidebar.slider("elev", 0, 360, 240, 1)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(gt)
        # plt.show()
        # ax.view_init(azim, elev)

        # st.pyplot()
        # m = Mesh.from_voxel_grid(
        #     voxels=pred.astype(bool),
        # )

        # show(
        #     Mesh.from_voxel_grid(voxels=pred.astype(bool)),
        #     light=(-1, -1, 1),
        
        # )