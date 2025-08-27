import numpy as np
import helper as hlp
import skimage.io as io
# import submission as sub
import matplotlib.pyplot as plt
from submission import eight_point, epipolar_correspondences, essential_matrix, triangulate

# 1. Load the two temple images and the points from data/some_corresp.npz
img1 = io.imread('data/im1.png')
img2 = io.imread('data/im2.png')
data = np.load('data/some_corresp.npz')

# 2. Run eight_point to compute F
M = np.max([img1.shape[0], img1.shape[1]])
pts1 = data['pts1']
pts2 = data['pts2']

F = eight_point(pts1, pts2, M)
# print("F matrix rank:", np.linalg.matrix_rank(F))
# hlp.displayEpipolarF(img1, img2, F)

# 3. Load points in image 1 from data/temple_coords.npz
pts1 = np.load('data/temple_coords.npz')['pts1']
# 4. Run epipolar_correspondences to get points in image 2
pts2 = epipolar_correspondences(img1, img2, F, pts1)
hlp.epipolarMatchGUI(img1, img2, F)

x = np.load('data/intrinsics.npz')
# print(x['K1'].shape)
# print(x['K2'].shape)
# 
# pts1 = np.load('data/temple_coords.npz')['pts1']
# pts2 = np.load('data/temple_coords.npz')
# print(pts1.shape)
# print(pts2)
# print(pts2.shape)

