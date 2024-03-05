"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import math
import numpy as np
import argparse
from einops import rearrange, repeat
# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from PIL import Image


class Model(nn.Module):
    def __init__(self, pcd, device):
        super(Model, self).__init__()
        self.device = device
        # set template mesh
        self.pcd = pcd.to(device) # [2048,3]
        self.register_buffer('init_colors', torch.zeros(2048))

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.init_colors)))

        # render
        self.views = [55,145,235,325] # degree of 360
        self.view_num = len(self.views)
        # self.R, self.T = look_at_view_transform(2.0, 10, 55) 
        self.R, self.T = look_at_view_transform(1.5, 15, self.views) 
        self.raster_settings = PointsRasterizationSettings(
            image_size=256, 
            radius = 0.01,
            points_per_pixel = 5
        )
        # self.cameras = PerspectiveCameras(R=self.R, T=self.T, device='cpu')
        self.cameras = FoVOrthographicCameras(device=device, R=self.R, T=self.T, znear=0.01)
        self.rasterizer=PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.renderer = PointsRenderer(
            rasterizer=self.rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color 
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=AlphaCompositor(background_color=(0, 0, 0))
        )

        self.linear1 = nn.Sequential(nn.Conv1d(256,256,1), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Conv1d(128,128,1), nn.ReLU())
        # self.linear3 = nn.Sequential(nn.Conv1d(1024,1,1))
        # self.linear4 = nn.Sequential(nn.Linear(6144,2048), nn.Sigmoid())
        self.attention1 = attention(2048, 3, 256)
        self.attention2 = attention(2048, 256, 128)
        self.attention3 = attention(2048, 128, 1)
        self.linear = nn.Linear(2048,2048)
        self.linear2 = nn.Linear(3*2048,2048)

    def forward(self):
        input_points = torch.permute(self.pcd, (1,0)) # (3,2048)
        input_points = input_points.unsqueeze(0) # (1,3,2048)

        # x = self.attention1(input_points)
        # x = self.attention2(x)
        # x = self.attention3(x)
        # x = rearrange(x,'B C N -> B (C N)')
        # x = self.linear(x)
        # x = F.sigmoid(x)
        # x = rearrange(x,'B N -> (B N)')

        x = rearrange(input_points, 'B C N -> B (C N)')
        x = self.linear2(x)
        x = F.sigmoid(x)
        x = rearrange(x,'B N -> (B N)')


        base = self.init_colors.to(self.device)
        colors = torch.sigmoid(base+self.displace) # [2048]
        colors = x

        points = self.pcd.unsqueeze(0).repeat_interleave(4, dim=0)
        colors = colors.unsqueeze(1).repeat(1,3) # (2048) -> (2048,3)
        colors = colors.unsqueeze(0).repeat_interleave(4, dim=0)

        # point_cloud = Pointclouds(points=[self.pcd], features=[colors])
        point_cloud = Pointclouds(points=[points[i] for i in range(points.shape[0])], 
                                features=[colors[i] for i in range(colors.shape[0])])
        images = self.renderer(point_cloud) # (4,256,256,3)
        return images

def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()

def save_img(img,file_name):
    # img (256,256,3) np array
    img = img*255
    img = (img).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)

def WeightL1(pred, target):
    pred = pred.sum(dim=-1)/3     # (B,4,256,256,3) -> (B,4,256,256)
    target = target.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256)
    # L1 loss
    # count_all = torch.sum(target > 0)
    # L1_loss = torch.abs(pred - target)
    # L1_loss = L1_loss.sum()/count_all

    # count_all = torch.sum((pred <= target) & (target > 0))
    count_all = torch.sum(target > 0)
    L1_loss = torch.where(pred <= target, (pred - target)*4, pred - target)
    # L1_loss = pred - target
    L1_loss = L1_loss.abs()
    L1_loss = L1_loss.sum()/count_all

    return L1_loss

class attention(nn.Module):
    def __init__(self, npts_ds = 2048, input_dim = 3, output_dim = 3):
        super(attention, self).__init__()
        self.npts_ds = npts_ds
        self.q_conv = nn.Conv1d(input_dim, output_dim, 1, bias=False)
        self.k_conv = nn.Conv1d(input_dim, output_dim, 1, bias=False)
        self.v_conv = nn.Conv1d(input_dim, output_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q_conv(x)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(x)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(x)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        out = attention @ rearrange(v, 'B C N -> B N C').contiguous() # (B, N, N) @ (B, N, C) -> (B, N, C)
        out = rearrange(out,'B N C -> B C N').contiguous()
        return out

def main():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # load pcd
    file_num = '0002'
    pcd_path = f'data/modelnet_small/pcd/train/{file_num}.npy'
    pcd = np.load(pcd_path)
    # 将点云平移到原点
    centroid = np.mean(pcd, axis=0)
    point_cloud_centered = pcd - centroid
    # 缩放点云使其适应[-1, 1]范围
    max_distance = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
    point_cloud_normalized = point_cloud_centered / max_distance
    pcd_tensor = torch.tensor(point_cloud_normalized).to(torch.float32)

    # load gt
    multi_view_paths = [f'data/modelnet_small/pcd/train/alphashape9*9_img/{file_num}_0.png',
                        f'data/modelnet_small/pcd/train/alphashape9*9_img/{file_num}_1.png',
                        f'data/modelnet_small/pcd/train/alphashape9*9_img/{file_num}_2.png',
                        f'data/modelnet_small/pcd/train/alphashape9*9_img/{file_num}_3.png']

    multi_view_imgs = [Image.open(path) for path in multi_view_paths]
    np_imgs = [np.array(img) for img in multi_view_imgs]
    gt_np = [img.astype(np.float32) / 255. for img in np_imgs]
    save_img(gt_np[0],'demo_results/gt.png')
    images_gt = torch.tensor(gt_np).to(device)
    # print("images_gt: ", images_gt.shape)

    model = Model(pcd_tensor, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    loss_fn = nn.MSELoss(reduction="mean")

    images = model.forward()

    loop = tqdm.tqdm(list(range(0, 51)))
    for i in loop:
        images = model.forward()

        loss = WeightL1(images, images_gt)
        # loss = F.l1_loss(images, images_gt)

        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            save_img(images[0].detach().cpu().numpy(),f'demo_results/output_{i}_0.png')
            save_img(images[1].detach().cpu().numpy(),f'demo_results/output_{i}_1.png')
            save_img(images[2].detach().cpu().numpy(),f'demo_results/output_{i}_2.png')
            save_img(images[3].detach().cpu().numpy(),f'demo_results/output_{i}_3.png')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()