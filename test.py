import numpy as np

x = np.load('../data/intrinsics.npz')
print(x['K1'].shape)
print(x['K2'].shape)

# pts1 = np.load('../data/temple_coords.npz')['pts1']
# pts2 = np.load('../data/temple_coords.npz')
# print(pts1.shape)
# print(pts2)
# print(pts2.shape)

