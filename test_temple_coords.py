import numpy as np
import helper as hlp
import skimage.io as io
# import submission as sub
import matplotlib.pyplot as plt
from submission import eight_point, epipolar_correspondences, essential_matrix, triangulate

# 1. Load the two temple images and the points from data/some_corresp.npz
img1 = io.imread('../data/im1.png')
img2 = io.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')

# 2. Run eight_point to compute F
M = np.max([img1.shape[0], img1.shape[1]])
pts1 = data['pts1']
pts2 = data['pts2']

F = eight_point(pts1, pts2, M)
# print("F matrix rank:", np.linalg.matrix_rank(F))
# hlp.displayEpipolarF(img1, img2, F)


# 3. Load points in image 1 from data/temple_coords.npz
pts1 = np.load('../data/temple_coords.npz')['pts1']
# 4. Run epipolar_correspondences to get points in image 2
pts2 = epipolar_correspondences(img1, img2, F, pts1)
# hlp.epipolarMatchGUI(img1, img2, F)

# 5. Compute the camera projection matrix P1
K1_extrinsic = np.hstack((np.eye(3, 3), np.zeros((3,1))))
## P = K[I|0]
#### Load camera intrinsic parameters 
K1_intrinsic, K2_intrinsic = np.load('../data/intrinsics.npz')['K1'], np.load('../data/intrinsics.npz')['K2']
P1 = K1_intrinsic @ K1_extrinsic

# 6. Use camera2 to get 4 camera projection matrices P2
#### get Essential matrix
E = essential_matrix(F, K1_intrinsic, K2_intrinsic)
## get all possible extrinsics for C2
M2_candidates = hlp.camera2(E) ## (3,4,4) 4 possible [R | t] combinations for the second camera

# 7. Run triangulate using the projection matrices
best_M2, best_P2 = None, None
best_pts3d = None
max_positive = 0

for i in range(4):
    M2 = M2_candidates[:, :, i] ## M2 is extrinsic candidate
    P2_candidate = K2_intrinsic @ M2

    ### get 3d points (X, Y, Z)
    pts3d_candidate = triangulate(P1, pts1, P2_candidate, pts2)

# 8. Figure out the correct P2
    ## convert the pts3d to homogenous
    pts_h = np.hstack([pts3d_candidate, np.ones((pts3d_candidate.shape[0], 1))])

    ### camera 1 depth, cuz for camera 1 P1 is [I|0]
    Z1 = pts_h[:, 2] ### get Z from [X, Y, Z, 1]

    ## camera 2 coords
    ### convert 3D point to camera point x = [R | T]X
    cam2_coords = ( M2 @ pts_h.T).T
    Z2 = cam2_coords[:, 2]

    ## counting how many points have Z1 and Z2 > 0 for P2 and M2 candidate
    positive_count = np.sum((Z1 > 0) & (Z2 > 0))
    if positive_count > max_positive:
        max_positive = positive_count
        best_M2 = M2
        best_P2 = P2_candidate
        best_pts3d = pts3d_candidate

# 9. Scatter plot the correct 3D points
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection ='3d')
ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2], s=5)
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
np.savez('../data/extrinsics.npz', R1 = np.eye(3), t1 = np.zeros(3), 
         R2 = best_M2[:, :3], t2 = best_M2[:, 3 ])