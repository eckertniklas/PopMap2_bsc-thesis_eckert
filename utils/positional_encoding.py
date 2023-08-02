import torch
import torch.nn as nn
import math

from utils.plot import plot_2dmatrix

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, channels=32):
        super(PositionalEncoding2D, self).__init__()
        h, w = dim
        self.h = h
        self.w = w
        self.max_len = channels

        # just create a vector and later expand it to 2d
        self.h_pe = torch.zeros(self.max_len*2, h)
        h_position = torch.arange(0, h)/ h *2* math.pi
        for i in range(self.max_len):
            self.h_pe[i, :] = torch.sin(h_position * i)
            self.h_pe[i+self.max_len, :] = torch.cos(h_position * i)

        # now for width
        self.w_pe = torch.zeros(self.max_len*2, w)
        w_position = torch.arange(0, w)/ w *2* math.pi
        for i in range(self.max_len):
            self.w_pe[i, :] = torch.sin(w_position * i)
            self.w_pe[i+self.max_len, :] = torch.cos(w_position * i)
            

    def forward(self, window):

        # window = ((new_x,x_stop),(new_y,y_stop))

        xemb = self.h_pe[:,window[0][0]:window[0][1]].unsqueeze(2).repeat(1,1,window[1][1]-window[1][0])
        yemb = self.w_pe[:,window[1][0]:window[1][1]].unsqueeze(1).repeat(1,window[0][1]-window[0][0],1)


        return torch.cat((xemb, yemb), dim=0)

