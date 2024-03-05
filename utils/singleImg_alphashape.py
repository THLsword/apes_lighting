import matplotlib.pyplot as plt
from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from alpha_shapes.boundary import Boundary, get_boundaries
from PIL import Image
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np




# 读取 PNG 图片
image_path = "data/modelnet/pcd/train/render_img/0000_0.png"  # 替换为你的图片路径
image = Image.open(image_path)

# 将图片转换为 numpy 数组
data = np.array(image)
print(data.shape[0])
print(data.shape[1])
# 提取所有颜色值小于 255 的点的坐标

points_2d=[]
for x in range(data.shape[1]):
    for y in range(data.shape[0]):
        if np.any(data[y, x] > 0):
            points_2d.append((x, data.shape[0]-1-y))

shaper = Alpha_Shaper(points_2d)
print(len(points_2d))
try:

    alpha = 15.0
    alpha_shape = shaper.get_shape(alpha=alpha)

    fig, ax = plt.subplots()
    ax.scatter(*zip(*points_2d))
    print(type(ax))
    # plot_alpha_shape(ax, alpha_shape)

    vertices = []
    for boundary in get_boundaries(alpha_shape):
        exterior = Path(boundary.exterior)
        holes = [Path(hole) for hole in boundary.holes]
        path = Path.make_compound_path(exterior, *holes)
        # print(type(path))
        # print(path)
        print(path.vertices.shape)
        vertices.append(path.vertices)

        patch = PathPatch(path, facecolor="r", lw=0.8, alpha=0.5, ec="b")
        newpath = patch.get_path()
        print(newpath.vertices.shape)
        ax.add_patch(patch)
except TypeError:
    alpha = 3.0
    alpha_shape = shaper.get_shape(alpha=alpha)

    fig, ax = plt.subplots()
    ax.scatter(*zip(*points_2d))
    print(type(ax))
    # plot_alpha_shape(ax, alpha_shape)

    vertices = []
    for boundary in get_boundaries(alpha_shape):
        exterior = Path(boundary.exterior)
        holes = [Path(hole) for hole in boundary.holes]
        path = Path.make_compound_path(exterior, *holes)
        # print(type(path))
        # print(path)
        print(path.vertices.shape)
        vertices.append(path.vertices)

        patch = PathPatch(path, facecolor="r", lw=0.8, alpha=0.5, ec="b")
        newpath = patch.get_path()
        print(newpath.vertices.shape)
        ax.add_patch(patch)


npvertices = np.concatenate(vertices)
print(npvertices.shape)

# plt.axis("off")
# plt.savefig("test-as.png")

img = Image.new('RGB', (256, 256), color='black')
pixels = img.load()

# 将数组中的每个点设置为白色
for point in npvertices:
    x, y = point
    # neighbors = [(x-2, y-2), (x-1, y-2), (x, y-2), (x+1, y-2), (x+2, y-2),
    #             (x-2, y-1), (x-1, y-1), (x, y-1), (x+1, y-1),(x+2, y-1),
    #             (x-2, y), (x-1, y),               (x+1, y), (x+2, y),
    #             (x-2, y+1), (x-1, y+1), (x, y+1), (x+1, y+1),(x+2, y+1),
    #             (x-2, y+2), (x-1, y+2), (x, y+2), (x+1, y+2), (x+2, y+2)]

    neighbors = [(x-1, y-1), (x, y-1), (x+1, y-1),
                 (x-1, y),             (x+1, y),
                 (x-1, y+1), (x, y+1), (x+1, y+1)]
    
    
    for nx, ny in neighbors:
        nx=int(nx)
        ny=int(ny)
        # Check boundaries
        if 0 <= nx < data.shape[0] and 0 <= ny < data.shape[1]:
            # print(data[nx, ny])
            # pixels[nx, data.shape[0]-1-ny] = (0, 0, 0)

            if np.any(data[data.shape[0]-1-ny, nx] > 0):
                pixels[nx, data.shape[0]-1-ny] = (255, 255, 255)
    pixels[x, data.shape[0]-1-y] = (255, 255, 255) 

img.save('test-as.png')

