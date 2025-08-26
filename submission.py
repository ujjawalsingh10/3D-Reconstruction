"""
Homework 5
Submission Functions
"""

import numpy as np
import helper
from scipy.signal import convolve2d
from scipy.linalg import inv 

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    ### NORMALIZATION
    T = np.array([[1/M, 0, 0],
                  [0, 1/M, 0,],
                  [0, 0, 1]])
    n = pts1.shape[0]
    ### homogeneous coords
    pts1_h = np.hstack([pts1, np.ones((n, 1))])
    pts2_h = np.hstack([pts2, np.ones((n, 1))])
    pts1_norm = T.dot(pts1_h.T).T
    pts2_norm = T.dot(pts2_h.T).T

    #### Construct A matrix
    ## 1 equation for 1 correspondance
    A = np.zeros((n, 9))
    for i in range(n):
        x, y = pts1_norm[i][0], pts1_norm[i][1]
        u, v = pts2_norm[i][0], pts2_norm[i][1]
        A[i] = [x * u, x * v, x, y * u, y *v, y, u, v, 1]

    ## solving using SVD Af = 0
    u1, s1, vh1 = np.linalg.svd(A)
    F_raw = vh1[-1, :].reshape(3, 3)  ## reshape the 9, 1 to 3x3
                    
    ### Enforce Rank 2 Constraint
    U, S, Vt = np.linalg.svd(F_raw)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    ## refine 
    F_ref = helper.refineF(F, pts1_norm[:,:2], pts2_norm[:, :2])

    ### unnormalize
    F_out = T.T @ F_ref @ T
    return F_out



"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    """
    pts1 → This is an N×2 matrix of points in the first image (e.g., [[120, 85], [200, 150], ...]).

    For each point in pts1, the algorithm:

    Finds the epipolar line in the second image using the fundamental matrix F.

    Searches along that line for the best matching point.

    Stores that best match as (x2, y2) in pts2.
    """
    ## 11 x 11 small matrix around pixels are compared to 
    pts2 = [] ## to get epipolar

    ###############
    window_size = 5
    w = window_size // 2 ## half window size for candidate comparison
    n = pts1.shape[0] ## total number of points
    pts1 = np.hstack([pts1, np.ones((n, 1))]) ## homogeneous coordinates

    for point in pts1:
        x1, y1 = point[0], point[1] ## used below for patch extraction from image1
        ## calc epipolar line in im2
        l = F @ point
        a, b, c = l ## ax1 + by1 + c = 0

        # Avoid division by zero if the line is nearly vertical
        if abs(b) < 1e-6:
            continue

        ## search along the line
        best_match = None
        best_score = float('inf')

        for x2 in range(w, im2.shape[1] - w):##shape[1] for x coords
            y2 = int(round((-a*x2 - c) / b))
            ##check whether is it going outside the image with window
            if y2 < w or y2 >= im2.shape[0] or y1 <w or y1 >= im1.shape[0]: ## shape[0] for y coords
                continue

            ## extract the patches
            patch1 = im1[int(y1-w) : int(y1+w+1), int(x1-w) : int(x1+w+1)]
            patch2 = im2[y2-w : y2+w+1, x2-w : x2+w+1]

            ## compare using ssd (sum of squared pixel differences)
            score = np.sum((patch1 - patch2) ** 2)

            ## smallest ssd is the match
            if score < best_score:
                best_score = score
                best_match = (x2, y2)        
        pts2.append(best_match)
    return np.array(pts2)

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    E = K2.T @ F @ K1 ###  k'T.F.K (K'= intrinsic for C2 amd K for C1)
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3d = np.zeros((N, 3))
    
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])

        # A = np.array([
        #     # x1 * P1[2, :] - P1[0, :],
        #     # y1 * P1[2, :] - P1[1, :],
        #     # x2 * P2[2, :] - P2[0, :],
        #     # y2 * P2[2, :] - P2[1, :]
        #     # y1 * P1[2, :] - P1[1, :],
        #     # P1[0, :] - x1 * P1[2, :],
        #     # y2 * P2[2, :] - P2[1, :],
        #     # P2[0, :] - x2 * P2[2, :]
        #     x1 * P1[2, :] - P1[1, :],
        #     P1[0, :] - y1 * P1[2, :],
        #     x2 * P2[2, :] - P2[1, :],
        #     P2[0, :] - y2 * P2[2, :]
        # ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        pts3d[i] = X[:3]
    return pts3d

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)

"""

def camera_center(K,R,t):
    A=-np.linalg.inv(np.matmul(K,R))
   
    B=np.matmul(K,t)
    result=np.matmul(A,B)
    return result

def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1= camera_center(K1,R1,t1)
    c1.shape=(1,3)
    c2= camera_center(K2,R2,t2)
    c2.shape=(1,3)

    r1=(c1-c2)/np.linalg.norm(c1-c2)
    r2=np.cross( r1, R1.T[:, -1]) 
    
    r3=np.cross(r2,r1)
    
    R=np.hstack((r1,r2))
    R=np.hstack((R,r3))
    R.shape=(3,3)
  
    R=np.transpose(R)
 

    t1p=-np.matmul(R,np.transpose(c1))
    t2p=-np.matmul(R,np.transpose(c2))


    M1=np.matmul(np.matmul(K2,R),inv(np.matmul(K1,R1)))
    M2=np.matmul(np.matmul(K2,R),inv(np.matmul(K2,R2)))
    return M1,M2,K2,K2,R,R,t1p,t2p

# def rectify_pair(K1, K2, R1, R2, t1, t2):
#     # replace pass by your implementation

#     ## optical center c1, c2 for both cam is calc by c = -R.T * t
#     c1 = -R1.T @ t1
#     c2 = -R2.T @ t2

#     #### calc the new rotation matrix
#     ## r1
#     baseline = c1 - c2
#     r1 = baseline / np.linalg.norm(baseline)

#     ## r2 
#     #cross product of old z axis and r1.. to get axis perpendicular (giving up direction)
#     # axis ortho to r1 
#     old_z = R1[2,:]
#     ortho_to_r1 = np.cross(old_z, r1)
#     norm_ortho_to_r1 = np.linalg.norm(ortho_to_r1)
#     if norm_ortho_to_r1 < 1e-18:
#         raise ValueError('Old Z is parallel to baseline')
#     r2 = ortho_to_r1 / norm_ortho_to_r1

#     ## r3 
#     ### perpendicular to r2 and r1 to get the new z axis
#     r3 = np.cross(r2, r1)

#     ##### the new rotation matrix
#     R = np.vstack([r1, r2, r3]).T

#     ## new K whicih is common and is gonna be used for same focal length, skewness and pp
#     K = K2.copy()

#     ## computing the new translation vectors
#     tp1 = -R @ c1
#     tp2 = -R @ c2

#     ## final rectification matrix of the cameras M1 and M2
#     M1 = K @ R @ np.linalg.inv(K1 @ R1)
#     M2 = K @ R @ np.linalg.inv(K2 @ R2)

#     ### rectified camera matrices
#     K1p = K.copy()
#     K2p = K.copy()

#     ### new Rotation matrix
#     R1p = R2p = R.copy(), R.copy()

#     return M1, M2, K1p, K2p, R1p, R2p, tp1, tp2



"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    H, W = im1.shape
    w = win_size // 2
    
    # kernel for windowed sum 
    kernel = np.ones((win_size, win_size))
    
    # cost volume (H,W,D)
    cost_vol = np.zeros((H, W, max_disp+1), dtype=np.float32)
    
    for d in range(max_disp+1):

        # shift right image by disparity d
        shifted = np.zeros_like(im2)
        if d > 0:
            shifted[:, d:] = im2[:, :-d]
        else:
            shifted = im2.copy()
            
        # squared difference
        diff2 = (im1 - shifted)**2
        
        # sum over window via convolution
        cost = convolve2d(diff2, kernel, mode='same', boundary='fill', fillvalue=0) ## assumes padding 0 for going out of bound
        
        ## for every pixel (y, x) store z in D for H,W,D vol 
        cost_vol[:,:,d] = cost
    
    # choose disparity with min cost
    dispM = np.argmin(cost_vol, axis=2)
    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    H, W = dispM.shape[0], dispM.shape[1]

    f = K1[0,0]
    B = np.linalg.norm(t1-t2)


    depthM = np.zeros_like(dispM)

    mask = dispM > 0 ## all values ==0 become false
    depthM[mask] = (B * f) / dispM[mask]

    # top = B * f
    # for i in range(H):
    #     for j in range(W):
    #         d = dispM[i,j]
    #         if d > 0:
    #             depthM[i,j] = top / d
    #         else:
    #             depthM[i,j] = 0
    return depthM
        



"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
