import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from typing import List
import os
import time
import sys
from einops import repeat, pack, rearrange
from typing import List, Dict
from pytorch_lightning import LightningModule
import numpy as np
import torch.optim.lr_scheduler as lrs
import matplotlib.pyplot as plt
from PIL import Image


# 创建一个简单的模型
model = nn.Linear(10, 1)

# 定义输入和目标
input = torch.randn(3, 10)
print(input.requires_grad)
target = torch.randn(3, 1)

# 定义MSELoss和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模式
model.train()

# 训练步骤
for epoch in range(100):  # 训练100个周期
    # 清除梯度
    optimizer.zero_grad()
    
    # 进行前向传播
    output = model(input)
    output.retain_grad()
    
    # 计算损失
    loss = criterion(output, target)
    
    # 反向传播
    loss.backward()
    # print(output.grad)

    
    # 更新参数
    optimizer.step()
    
    # 打印损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item()}')