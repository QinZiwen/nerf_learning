import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm, trange

np.random.seed(0)

splits = ['train', 'val', 'test']
basedir = '../dataset/nerf_synthetic/lego'

metas = {}
for s in splits:
    with open(os.path.join(basedir, 'transforms_{}.json'.format(s))) as fp:
        metas[s] = json.load(fp)
        
all_images = []
all_poses = []
counts = [0]

for s in splits:
    meta = metas[s]
    imgs = []
    poses = []
    for frame in meta['frames'][::]:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))
        # print('fname: ', fname)
        
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    counts.append(counts[-1] + imgs.shape[0])
    all_images.append(imgs)
    all_poses.append(poses)

imgs = np.concatenate(all_images, axis=0)
poses = np.concatenate(all_poses, axis=0)

print("after concatenate, imgs.shape: ", imgs.shape)
print("after concatenate, poses.shape: ", poses.shape)

plt.figure('lego')
plt.imshow(imgs[0])
plt.title('lege image')
plt.show()