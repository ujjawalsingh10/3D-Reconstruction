import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import helper as hlp
import submission as sub
import numpy.linalg as la
import skimage.color as col
import matplotlib.pyplot as plt

# 1. Load the images and the parameters

I1 = cv.cvtColor(cv.imread('data/im1.png'), cv.COLOR_BGR2GRAY).astype(np.float32)
I2 = cv.cvtColor(cv.imread('data/im2.png'), cv.COLOR_BGR2GRAY).astype(np.float32)

rectify = np.load('data/rectify.npz')
M1, M2 = rectify['M1'], rectify['M2']
K1p, K2p = rectify['K1p'], rectify['K2p']
R1p, R2p = rectify['R1p'], rectify['R2p']
t1p, t2p = rectify['t1p'], rectify['t2p']

# 2. Get disparity and depth maps

max_disp, win_size = 20, 3
dispM = sub.get_disparity(I1, I2, max_disp, win_size)
depthM = sub.get_depth(dispM, K1p, K2p, R1p, R2p, t1p, t2p)

def depth_to_pointcloud(depthM, K, rgb=None, max_points=100000):
    """
    Convert depth map into 3D point cloud.

    depthM : (H, W) depth map
    K      : camera intrinsic matrix (3x3)
    rgb    : optional (H, W, 3) color image for coloring points
    max_points : limit to avoid slow plotting
    """
    H, W = depthM.shape
    i, j = np.indices((H, W))  # pixel coordinates

    # Unproject pixels to camera coordinates
    Z = depthM
    X = (j - K[0, 2]) * Z / K[0, 0]
    Y = (i - K[1, 2]) * Z / K[1, 1]

    # Flatten
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Filter out invalid depths
    mask = Z > 0
    X, Y, Z = X[mask], Y[mask], Z[mask]

    if rgb is not None:
        colors = rgb.reshape(-1, 3)[mask] / 255.0
    else:
        colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (len(X), 1))  # gray

    # Randomly sample for speed
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X, Y, Z, colors = X[idx], Y[idx], Z[idx], colors[idx]

    return X, Y, Z, colors

# Example usage:
# depthM = your depth map (H, W)
# I1 = grayscale or color image (H, W, 3)
rgb=cv.imread(('data/im1.png'), cv.COLOR_BGR2RGB)
X, Y, Z, colors = depth_to_pointcloud(depthM, K1p, rgb=rgb)

# Plot 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c=colors, s=0.5)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Point Cloud from Stereo Depth')
plt.show()
