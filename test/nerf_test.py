import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm, trange

import utils

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

i = 0
i_train = []
i_val = []
i_test = []
for s in splits:
    meta = metas[s]
    imgs = []
    poses = []
    for frame in meta['frames'][::]:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        img = imageio.imread(fname)
        if img.shape[-1] == 4:  # 检查是否有 Alpha 通道
            img = img[:, :, :3]  # 只保留前三个通道（RGB）
        # print('img.shape: ', img.shape)
        imgs.append(img)
        poses.append(np.array(frame['transform_matrix']))
        # print('fname: ', fname)
        
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    counts.append(counts[-1] + imgs.shape[0])
    all_images.append(imgs)
    all_poses.append(poses)
    
    if s == 'train':
        i_train = np.arange(counts[-2], counts[-1])
    elif s == 'val':
        i_val = np.arange(counts[-2], counts[-1])
    elif s == 'test':
        i_test = np.arange(counts[-2], counts[-1])

print("counts: ", counts)
# print("i_train: ", i_train, " i_val: ", i_val, " i_test: ", i_test)
print("before concatenate, imgs.shape: ", len(all_images))
print("before concatenate, poses.shape: ", len(all_poses))
imgs = np.concatenate(all_images, axis=0)
poses = np.concatenate(all_poses, axis=0)
print("after concatenate, imgs.shape: ", imgs.shape)
print("after concatenate, poses.shape: ", poses.shape)

# plt.figure('lego')
# plt.imshow(imgs[0])
# plt.title('lege image')
# plt.show()

camera_angle_x = float(meta['camera_angle_x'])
H, W = imgs[0].shape[:2]

focal = .5 * W / np.tan(.5 * camera_angle_x)
K = np.array([[focal, 0, 0.5 * W],
              [0, focal, 0.5 * H],
              [0, 0, 1]])

for p in poses[:, :3, :4]:
    print(p)
    break
rays = np.stack([utils.get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
print('rays: ', rays.shape)

print("imgs[:, None]]: ", imgs[:, None].shape)
rays_rgb = np.concatenate([rays, imgs[:, None]], 1)
print('rays_rgb: ', rays_rgb.shape)
rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
print('np.transpose rays_rgb: ', rays_rgb.shape, rays_rgb[0].shape)

rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
print('train rays_rgb: ', rays_rgb.shape)

rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
print('reshape rays_rgb: ', rays_rgb.shape)

rays_rgb = rays_rgb.astype(np.float32)
np.random.shuffle(rays_rgb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# imgs = torch.Tensor(imgs).to(device)
# poses = torch.Tensor(poses).to(device)
rays_rgb = torch.Tensor(rays_rgb).to(device)

