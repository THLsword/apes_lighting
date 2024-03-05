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

A = torch.randint(5,(2,3,5))
print(A)
B = A.reshape(2,5,3)
print(B)
C = rearrange(A,'B C N -> B N C')
print(C)