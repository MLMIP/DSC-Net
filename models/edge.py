import torch
import torch.nn as nn

class edge(nn.Module):
    def __int__(self):
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
    def layers(self,input):
        x1 = self.conv1(input)

        p1 = self.pool(x1)

        sub = x1 - p1 # 边界

        add = sub + x1 # 主体

        return sub, add