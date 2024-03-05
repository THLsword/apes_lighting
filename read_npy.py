import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
# 读取.npy文件
file_path = 'data/shapenet/seg_label/train/0007.npy'  # 替换为你的文件路径
data = np.load(file_path)

# 打印内容
print(data)
