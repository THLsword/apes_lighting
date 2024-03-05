import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

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

from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from alpha_shapes.boundary import Boundary, get_boundaries
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tqdm import tqdm



def render_pcd(DATA_DIR,SAVE_DIR,file_path,filename):
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load point cloud
    pointcloud = np.load(file_path)
    # 将点云平移到原点
    centroid = np.mean(pointcloud, axis=0) 
    point_cloud_centered = pointcloud - centroid
    # 缩放点云使其适应[-1, 1]范围
    max_distance = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
    point_cloud_normalized = point_cloud_centered / max_distance

    verts = torch.Tensor(point_cloud_normalized).to(device)
    rgb = torch.ones_like(verts).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    # Initialize a camera.
    list_R=[]
    list_T=[]
    for x in [55,145,235,325]:
        temp_R, temp_T = look_at_view_transform(1.5, 15, x)
        list_R.append(temp_R)
        list_T.append(temp_T)

    raster_settings = PointsRasterizationSettings(
        image_size=256, 
        radius = 0.01,
        points_per_pixel = 5
    )

    for i in range(len(list_R)):
        # camera_list.append(PerspectiveCameras(R=list_R[i], T=list_T[i], device=device))
        # cameras = PerspectiveCameras(R=list_R[i], T=list_T[i], device=device)
        cameras = FoVOrthographicCameras(device=device, R=list_R[i], T=list_T[i], znear=0.01)
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color 
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0))
        )
        images = renderer(point_cloud)
        image_data = images[0, ..., :3].cpu().numpy()
        image_data = (image_data * 255).astype(np.uint8)

        image = Image.fromarray(image_data)
        SAVE_filename=f'{os.path.splitext(filename)[0]}_{i}.png'
        image.save(os.path.join(SAVE_DIR,SAVE_filename))

# Set paths
DATA_DIR = "./data/modelnet/pcd/train"
SAVE_DIR = "./utils"
filename = "0368.npy"

file_path = os.path.join(DATA_DIR, filename)
render_pcd(DATA_DIR,SAVE_DIR,file_path,filename)