import numpy as np
from collections import OrderedDict
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
 
class DynamicFilterLayer(nn.Module): #MergeLayer
    def __init__(self, filter_size, stride=(1,1), pad=(0,0), flip_filters=False, grouping=False):
        super(DynamicFilterLayer, self).__init__()
        self.filter_size = filter_size #tuple 3
        self.stride = stride           #tuple 2
        self.pad = pad                 #tuple 2
        self.flip_filters = flip_filters
        self.grouping = grouping
 
    def get_output_shape_for(self, input_shapes):
        if self.grouping:
            shape = (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2], input_shapes[0][3])
        else:
            shape = (input_shape[0][0], 1, input_shapes[0][2], input_shapes[0][3])
        return shape
 
    def forward(self, _input, **kwargs):
    #def get_output_for(self, _input, **kwargs):
        image = _input[0]
        filters = _input[1]

 
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)
        filter_size = self.filter_size
 

        if self.grouping:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (np.prod(filter_size),  filter_size[0], filter_size[1]))
            filter_localexpand = filter_localexpand_np.float() 
             
            outputs = []
             
            for i in range(3):
                input_localexpand = F.Conv2d(image[:, [i], :, :], kerns= filter_localexpand, 
                    subsample=self.stride, border_mode=border_mode, conv_mod= conv_mode)
                output = torch.sum(input_localexpand*filters[i], dim=1, keepdim=True)
                outputs.append(output)
                 
            output = torch.cat(outputs, dim=1)
         
        else:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size)), (np.prod(filter_size), filter_size[2], filter_size[0], filter_size[1]))
            filter_localexpand = torch.from_numpy(filter_localexpand_np.astype('float32')).cuda()
            input_localexpand = F.conv2d(image, filter_localexpand, padding = self.pad)
            output = torch.sum(input_localexpand*filters, dim=1, keepdim=True)
             
        return output