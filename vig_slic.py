# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import numpy as np
from gcn_lib import Grapher, act_layer
from timm_modified.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm_modified.models.helpers import load_pretrained
from timm_modified.models.layers import DropPath, to_2tuple, trunc_normal_
from timm_modified.models.registry import register_model

from fast_slic import Slic
from PIL import Image
import pdb
import PIL
import matplotlib.pyplot as plt
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'gnn_patch16_64': _cfg(
        crop_pct=0.9, input_size=(3, 64, 64),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

# DATA_MEAN = (0.5, 0.5, 0.5)
# DATA_STD = (0.5, 0.5, 0.5)
DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x



class Stem_tiny(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x
from skimage.segmentation import mark_boundaries
from scipy import ndimage
def segmapPrint(seg_map, filename):
    seg_map = seg_map.detach().cpu().numpy()
    dilation = ndimage.binary_dilation(seg_map)
    boundaries = seg_map != dilation

    # Calculate the centroid of each cluster
    centroids = ndimage.measurements.center_of_mass(seg_map, seg_map, range(196))

    # Create a plot
    plt.figure(figsize=(10, 10))

    # Show the segmentation map
    plt.imshow(seg_map, cmap='nipy_spectral')

    # Overlay the boundaries
    plt.imshow(boundaries, cmap='gray', alpha=0.5)

    # Add text at each centroid with the cluster index
    for i, (y, x) in enumerate(centroids):
        if not np.isnan(x) and not np.isnan(y):  # Ignore clusters that are not present in the map
            plt.text(x, y, str(i), color='black', fontsize=12, ha='center', va='center')

    # Save the figure
    plt.savefig(filename)
    
class Stem_slic_V2(nn.Module):
    # V2: Incorporates the convolutional grid features via slic-pooling, adapted to 14 by 14 nodes
    def __init__(self, img_size=224, in_dim=11, out_dim=768, numRows=14, compactness=10, act='relu'):
        super().__init__()
        self.numRows = numRows
        self.numSegments = numRows*numRows
        self.temp_imgsize = 56
        self.temp_windowsize = int(self.temp_imgsize / numRows)
        self.featDim = out_dim
        # 14 by 14에 맞춰짐
        self.convs_img = nn.Sequential(
            nn.Conv2d(3, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
        )
        self.slic = Slic(num_components=self.numSegments, compactness=compactness)
        self.pos_embed = nn.Parameter(torch.zeros(1, out_dim, 14, 14))

    def sort_segmap(self, seg_maps):
        # Create a grid representing pixel coordinates
        pdb.set_trace()
        # Convert seg_maps to long if it's not already
        seg_maps = seg_maps.long()

        # Create a grid representing pixel coordinates
        coord_grid = torch.stack(torch.meshgrid(torch.arange(64), torch.arange(64)), 0).float().to(seg_maps.device)

        # Expand segmentation maps to have a channel dimension of size 196 using one-hot encoding
        one_hot = torch.nn.functional.one_hot(seg_maps, num_classes=196).permute(0, 3, 1, 2).float()

        # Compute the centroids
        sum_one_hot = one_hot.sum(dim=[2, 3], keepdim=True)
        centroids = (one_hot.unsqueeze(2) * coord_grid.unsqueeze(0)).sum(dim=[3, 4]) / sum_one_hot

        # Flatten the centroids for sorting
        centroids_flatten = centroids.view(centroids.shape[0], centroids.shape[1], -1)

        # Sort the clusters based on their centroids in a row-major order
        sorted_indices = centroids_flatten.argsort(dim=-1)

        # Create a tensor that maps old labels to new labels based on the sorted order
        mapping = torch.arange(196, device=seg_maps.device).unsqueeze(0).unsqueeze(-1).expand(sorted_indices.shape)
        mapping = mapping.clone().scatter_(2, sorted_indices, mapping)

        # Apply the mapping to get the new segmentation map
        new_seg_maps = mapping.gather(2, seg_maps).squeeze(-1)
        
        
        
        # coord_grid = torch.stack(torch.meshgrid(torch.arange(64), torch.arange(64)), 0).float().to(seg_maps.device)

        # # Expand segmentation maps to have a channel dimension of size 196 using one-hot encoding
        # one_hot = torch.nn.functional.one_hot(seg_maps.long(), num_classes=196).permute(0, 3, 1, 2).float()

        # # Compute the centroids
        # sum_one_hot = one_hot.sum(dim=[2, 3], keepdim=True)
        # centroids = (one_hot.unsqueeze(2) * coord_grid.unsqueeze(0)).sum(dim=[3, 4]) / sum_one_hot

        # # Flatten the centroids for sorting
        # centroids_flatten = centroids.view(centroids.shape[0], centroids.shape[1], -1)

        # # Sort the clusters based on their centroids in a row-major order
        # _, sorted_indices = centroids_flatten.sort(dim=-1)

        # # Create a tensor that maps old labels to new labels based on the sorted order
        # mapping = torch.zeros_like(sorted_indices)
        # mapping.scatter_(1, sorted_indices, torch.arange(196, device=seg_maps.device).expand_as(sorted_indices))

        # # Apply the mapping to get the new segmentation map
        # new_seg_maps = mapping.gather(1, seg_maps.unsqueeze(1)).squeeze(1)

        # Apply the mapping to get the new segmentation map
        return new_seg_maps
        
    def compute_features(self, segmaps, imgs_convFeat):
        # TODO: maybe sort segmap? 
        # segmapPrint(segmaps[0],'0.png')
        # segmaps = self.sort_segmap(segmaps)
        # segmapPrint(segmaps[0],'1.png')
        # pdb.set_trace()
        # Get the unique labels in the segmentation map
        featDim = imgs_convFeat.shape[1]
        batchSize = segmaps.shape[0]
        labels = torch.arange(self.numSegments).unsqueeze(0).repeat(batchSize, 1).cuda()
        # labels = segmaps.flatten(1).unique(dim=1)
        
        max_num_labels = labels.shape[1]
        # Create meshgrid to compute coordinates
        x_range = torch.arange(segmaps.shape[2], device=segmaps.device)
        y_range = torch.arange(segmaps.shape[1], device=segmaps.device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range)

        # Expand dims for broadcasting
        segmaps_exp = segmaps.view(segmaps.shape[0], segmaps.shape[1], segmaps.shape[2], 1)
        x_grid_exp = x_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)
        y_grid_exp = y_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)

        # Compute masks for each label
        masks = (segmaps_exp == labels.view(segmaps.shape[0], 1, 1, max_num_labels) )
        masks = masks.float()
        masks_resized = torch.nn.functional.interpolate(masks.permute(0,3,1,2), size=[self.temp_imgsize, self.temp_imgsize], mode='bilinear')
        masks, masks_resized = masks.bool(), masks_resized.permute(0,2,3,1)>0.1
        # pdb.set_trace()
        
        # indexing feature vector for average pooling
        num_windows = (self.temp_imgsize // self.temp_windowsize)
        window_idxs = torch.arange(num_windows ** 2).view(num_windows, num_windows).cuda()
        imgs_idxs = window_idxs.repeat_interleave(self.temp_windowsize, dim=0).repeat_interleave(self.temp_windowsize, dim=1)
        imgs_convFeat_idx = imgs_idxs.unsqueeze(0).repeat(batchSize, 1, 1)
        imgs_convFeat_idx_ = imgs_convFeat_idx.unsqueeze(1).expand(-1, featDim, -1, -1)
        imgs_convFeat_idx_ = imgs_convFeat_idx_.flatten(-2, -1)   # cluster idx into 1D
        # pdb.set_trace()
        idxed_feat = torch.gather(imgs_convFeat.flatten(-2,-1), 2, imgs_convFeat_idx_).unsqueeze(-1).view(batchSize, featDim, self.temp_imgsize, -1)
        
        # average pooling idxed_feat to construct output feature tensor
        
        # idxed_feat = idxed_feat.unsqueeze(-1).expand(-1, -1, -1, -1, self.numSegments)      # numSegments로 expand 하는게 잘못된듯?
        # idxed_feat = (idxed_feat * masks_resized.unsqueeze(1).float()).sum(dim=(2, 3)) 
        
        masks_resized = masks_resized.permute(0,3,1,2).float()
        idxed_feat_ = torch.einsum('bfhw,bmhw->bfm', idxed_feat, masks_resized)
        masks_resized = masks_resized.sum(dim=(2,3))
        idxed_feat_ = (idxed_feat_/masks_resized.unsqueeze(1)).unsqueeze(-1).view(batchSize, featDim, self.numRows, -1)
        idxed_feat_ = torch.nan_to_num(idxed_feat_)
        return idxed_feat_
    
    def compute_features_V2(self, segmaps, imgs_convFeat):
        # V2: return max count idx feat
        featDim = imgs_convFeat.shape[1]
        batchSize = segmaps.shape[0]
        labels = torch.arange(self.numSegments).unsqueeze(0).repeat(batchSize, 1).cuda()
        # labels = segmaps.flatten(1).unique(dim=1)
        
        max_num_labels = labels.shape[1]
        # Create meshgrid to compute coordinates
        x_range = torch.arange(segmaps.shape[2], device=segmaps.device)
        y_range = torch.arange(segmaps.shape[1], device=segmaps.device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range)

        # Expand dims for broadcasting
        segmaps_exp = segmaps.view(segmaps.shape[0], segmaps.shape[1], segmaps.shape[2], 1)
        x_grid_exp = x_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)
        y_grid_exp = y_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)

        # Compute masks for each label
        masks = (segmaps_exp == labels.view(segmaps.shape[0], 1, 1, max_num_labels) )
        masks = masks.float()
        masks = torch.nn.functional.interpolate(masks.permute(0,3,1,2), size=[self.temp_imgsize, self.temp_imgsize], mode='bilinear')
        masks = masks.permute(0,2,3,1)>0.1
        
        # indexing feature vector for average pooling
        num_windows = (self.temp_imgsize // self.temp_windowsize)
        window_idxs = torch.arange(num_windows ** 2).view(num_windows, num_windows).cuda()
        imgs_idxs = window_idxs.repeat_interleave(self.temp_windowsize, dim=0).repeat_interleave(self.temp_windowsize, dim=1)
        
        flat_masks = masks.flatten(1,2)
        flat_idxs = imgs_idxs.flatten(0,1).unsqueeze(-1).unsqueeze(0).expand(flat_masks.shape) +1
        counts = torch.zeros(flat_masks.shape).long().cuda()
        # counts.scatter_add_(1, flat_masks.long(), flat_idxs.long())
        counts.scatter_add_(1, flat_idxs.long(), flat_masks.long())
        counts=(counts.argmax(1)-1).unsqueeze(1).repeat(1,self.featDim,1)
        convFeat_zeros = torch.zeros(imgs_convFeat.shape[0], self.featDim, 1).float().cuda()
        imgs_convFeat = torch.cat((convFeat_zeros, imgs_convFeat.flatten(-2,-1)), dim=-1)
        # imgs_convFeat = torch.cat((convFeat_zeros, imgs_convFeat), dim=1).flatten(-2,-1)
        idxed_feat = torch.gather(imgs_convFeat, 2, (counts+1))
        idxed_feat = idxed_feat.view(idxed_feat.shape[0], self.featDim, self.numRows, self.numRows)
        return idxed_feat
    
    def slic_init(self, org_x, x_convFeat):
        batchSize = len(x_convFeat)
        if torch.is_tensor(org_x):
            org_x = org_x.detach().cpu().numpy().astype('uint8')
        sliced = np.zeros((batchSize, org_x.shape[2], org_x.shape[2])).astype('int16')
        if org_x.shape[-1] != 3:
            org_x=org_x.transpose(0,2,3,1)
        for batchIdx in range(batchSize):
            sliced[batchIdx] = self.slic.iterate(org_x[batchIdx].copy(order='C'))      # Iterate required input: H X W X 3
            # foo = self.slic.iterate(org_x[batchIdx].copy(order='C'))
            # marked_img = mark_boundaries(org_x[batchIdx].copy(order='C'), foo)
            # plt.imsave('fuck_original.png', org_x[batchIdx].copy(order='C'))
            # plt.imsave('fuck.png', marked_img)
        
        sliced = torch.tensor(sliced).cuda() 
        # x: image, sliced: SLIC segmentation map, x_convFeat: convolutional features 
        x_out = self.compute_features(sliced, x_convFeat)       
        
        return sliced.clone(), x_out

    def forward(self, x, org_x):
        x_convFeat = self.convs_img(x) + self.pos_embed       # output: x = Batch, Out_dim, num_rows, num_rows
        # pdb.set_trace()
        x_slic, x = self.slic_init(org_x, x_convFeat)        # x_slic: B, 64, 64 / x: B, feat_dim, 14, 14
        # pdb.set_trace()
        # x = self.convs_slic(x)
        return x_slic, x, x_convFeat
    
class Stem_slic_V2_1(nn.Module):
    # V2: Incorporates the convolutional grid features via slic-pooling, adapted to 14 by 14 nodes
    def __init__(self, img_size=224, in_dim=11, out_dim=768, numRows=10, compactness=10, act='relu'):
        super().__init__()
        self.numRows = numRows
        self.numSegments = numRows*numRows
        self.temp_imgsize = 56
        self.temp_windowsize = int(self.temp_imgsize / numRows)
        # 14 by 14에 맞춰짐
        self.convs_img = nn.Sequential(
            nn.Conv2d(3, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
        )
        self.slic = Slic(num_components=self.numSegments, compactness=compactness)
        self.pos_embed = nn.Parameter(torch.zeros(1, out_dim, 14, 14))
        
    def compute_features(self, segmaps, imgs_convFeat):
        # TODO: maybe sort segmap? 
        # segmapPrint(segmaps[0],'0.png')
        # segmaps = self.sort_segmap(segmaps)
        # segmapPrint(segmaps[0],'1.png')
        # pdb.set_trace()
        # Get the unique labels in the segmentation map
        featDim = imgs_convFeat.shape[1]
        batchSize = segmaps.shape[0]
        labels = torch.arange(self.numSegments).unsqueeze(0).repeat(batchSize, 1).cuda()
        # labels = segmaps.flatten(1).unique(dim=1)
        
        max_num_labels = labels.shape[1]
        # Create meshgrid to compute coordinates
        x_range = torch.arange(segmaps.shape[2], device=segmaps.device)
        y_range = torch.arange(segmaps.shape[1], device=segmaps.device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range)

        # Expand dims for broadcasting
        segmaps_exp = segmaps.view(segmaps.shape[0], segmaps.shape[1], segmaps.shape[2], 1)
        x_grid_exp = x_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)
        y_grid_exp = y_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)

        # Compute masks for each label
        masks = (segmaps_exp == labels.view(segmaps.shape[0], 1, 1, max_num_labels) )
        masks = masks.float()
        masks_resized = torch.nn.functional.interpolate(masks.permute(0,3,1,2), size=[self.temp_imgsize, self.temp_imgsize], mode='bilinear')
        masks, masks_resized = masks.bool(), masks_resized.permute(0,2,3,1)>0.1
        # pdb.set_trace()
        
        # indexing feature vector for average pooling
        num_windows = (self.temp_imgsize // self.temp_windowsize)
        window_idxs = torch.arange(num_windows ** 2).view(num_windows, num_windows).cuda()
        imgs_idxs = window_idxs.repeat_interleave(self.temp_windowsize, dim=0).repeat_interleave(self.temp_windowsize, dim=1)
        imgs_convFeat_idx = imgs_idxs.unsqueeze(0).repeat(batchSize, 1, 1)
        imgs_convFeat_idx = imgs_convFeat_idx.unsqueeze(1).expand(-1, featDim, -1, -1)
        imgs_convFeat_idx = imgs_convFeat_idx.flatten(-2, -1)   # cluster idx into 1D
        pdb.set_trace()
        idxed_feat = torch.gather(imgs_convFeat.flatten(-2,-1), 2, imgs_convFeat_idx).unsqueeze(-1).view(batchSize, featDim, self.temp_imgsize, -1)
        
        # average pooling idxed_feat to construct output feature tensor
        
        # idxed_feat = idxed_feat.unsqueeze(-1).expand(-1, -1, -1, -1, self.numSegments)      # numSegments로 expand 하는게 잘못된듯?
        # idxed_feat = (idxed_feat * masks_resized.unsqueeze(1).float()).sum(dim=(2, 3)) 
        
        masks_resized = masks_resized.permute(0,3,1,2).float()
        idxed_feat = torch.einsum('bfhw,bmhw->bfm', idxed_feat, masks_resized)
        masks_resized = masks_resized.sum(dim=(2,3))
        idxed_feat = (idxed_feat/masks_resized.unsqueeze(1)).unsqueeze(-1).view(batchSize, featDim, self.numRows, -1)
        idxed_feat = torch.nan_to_num(idxed_feat)
        return idxed_feat
    
    def slic_init(self, org_x, x_convFeat):
        batchSize = len(x_convFeat)
        if torch.is_tensor(org_x):
            org_x = org_x.detach().cpu().numpy().astype('uint8')
        sliced = np.zeros((batchSize, org_x.shape[2], org_x.shape[2])).astype('int16')
        if org_x.shape[-1] != 3:
            org_x=org_x.transpose(0,2,3,1)
        for batchIdx in range(batchSize):
            sliced[batchIdx] = self.slic.iterate(org_x[batchIdx].copy(order='C'))      # Iterate required input: H X W X 3
            # foo = self.slic.iterate(org_x[batchIdx].copy(order='C'))
            # marked_img = mark_boundaries(org_x[batchIdx].copy(order='C'), foo)
            # plt.imsave('fuck_original.png', org_x[batchIdx].copy(order='C'))
            # plt.imsave('fuck.png', marked_img)
        
        sliced = torch.tensor(sliced).cuda() 
        # x: image, sliced: SLIC segmentation map, x_convFeat: convolutional features 
        x_out = self.compute_features(sliced, x_convFeat)        # TODO: slic에 따라서 pooling 해서 feature 계산
        
        return sliced.clone(), x_out

    def forward(self, x, org_x):
        x_convFeat = self.convs_img(x) + self.pos_embed       # output: x = Batch, Out_dim, num_rows, num_rows
        # pdb.set_trace()
        x_slic, x = self.slic_init(org_x, x_convFeat)        # x_slic: B, 64, 64 / x: B, feat_dim, 14, 14
        # pdb.set_trace()
        # x = self.convs_slic(x)
        return x_slic, x, x_convFeat

class Stem_slic(nn.Module):
    def __init__(self, img_size=224, in_dim=11, out_dim=768, numRows=14, compactness=10, act='relu'):
        super().__init__()
        self.numRows = numRows
        self.numSegments = numRows*numRows
        # self.mlp1 = nn.Sequential(nn.Linear(in_dim, out_dim//8))
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
        )
        self.slic = Slic(num_components=self.numSegments, compactness=compactness)    # TODO: SLIC 하는걸 여기 말고 dataloader에 추가?
        
    def compute_features(self, imgs, segmaps):
        # Get the unique labels in the segmentation map
        batchSize = imgs.shape[0]
        imgs = imgs.permute(0,2,3,1)
        labels = torch.arange(self.numSegments).unsqueeze(0).repeat(batchSize, 1).cuda()
        # labels = segmaps.flatten(1).unique(dim=1)
        
        max_num_labels = labels.shape[1]
        # Create meshgrid to compute coordinates
        x_range = torch.arange(segmaps.shape[2], device=segmaps.device)
        y_range = torch.arange(segmaps.shape[1], device=segmaps.device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range)

        # Expand dims for broadcasting
        segmaps_exp = segmaps.view(segmaps.shape[0], segmaps.shape[1], segmaps.shape[2], 1)
        x_grid_exp = x_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)
        y_grid_exp = y_grid.view(1, segmaps.shape[1], segmaps.shape[2], 1)

        # Compute masks for each label
        masks = segmaps_exp == labels.view(segmaps.shape[0], 1, 1, max_num_labels) 
        # Compute centroids
        
        x_centroids, y_centroids = (x_grid_exp * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2)), (y_grid_exp * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))
        nanIdxs = torch.isnan(x_centroids)
        x_centroids = torch.nan_to_num(x_centroids)
        y_centroids = torch.nan_to_num(y_centroids)
        
        x_centroids_rounded = torch.round(x_centroids).long()
        y_centroids_rounded = torch.round(y_centroids).long()
        
        # Compute image mean, std
        centroids = torch.stack((x_centroids, y_centroids), dim=-1) 
        slic_means = torch.stack( [(imgs[:,:,:,0].unsqueeze(-1) * masks).mean(dim=(1, 2)), (imgs[:,:,:,0].unsqueeze(-1) * masks).mean(dim=(1, 2)), (imgs[:,:,:,0].unsqueeze(-1) * masks).mean(dim=(1, 2))], dim=-1)
        slic_stds = torch.stack( [(imgs[:,:,:,0].unsqueeze(-1) * masks).std(dim=(1, 2)), (imgs[:,:,:,0].unsqueeze(-1) * masks).std(dim=(1, 2)), (imgs[:,:,:,0].unsqueeze(-1) * masks).std(dim=(1, 2))], dim=-1)
        
        batch_indices = torch.arange(segmaps.shape[0], device=segmaps.device).view(-1, 1).expand(-1, max_num_labels)
        rgb_centroids = imgs[batch_indices, y_centroids_rounded, x_centroids_rounded, :]
        rgb_centroids[nanIdxs] = 0
        out = torch.cat((centroids, slic_means, slic_stds, rgb_centroids), dim=-1) 
        featDim = out.shape[-1]
        out = out.permute(0,2,1).view(batchSize, featDim, self.numRows, self.numRows)
        return out
    
    def slic_init(self, x, org_x):
        batchSize = len(x)
        if torch.is_tensor(org_x):
            org_x = org_x.detach().cpu().numpy().astype('uint8')
        sliced = np.zeros((batchSize, x.shape[2], x.shape[3])).astype('int16')
        
        if org_x.shape[-1] != 3:
            org_x=org_x.transpose(0,2,3,1)
        for batchIdx in range(batchSize):
            sliced[batchIdx] = self.slic.iterate(org_x[batchIdx].copy(order='C'))      # Iterate required input: H X W X 3

        x, sliced = torch.tensor(x).cuda(), torch.tensor(sliced).cuda()
        x_out = self.compute_features(x, sliced)
        
        return sliced.clone(), x_out

    def forward(self, x, org_x):
        x_slic, x = self.slic_init(x, org_x)        # x_slic: B, 64, 64 / x: B, 11, 14, 14
        x = self.convs(x)
        return x_slic, x
    

class DeepGCN_slic_V3(torch.nn.Module):
    # V2: Samples square window features based on SLIC assignment
    # V3: Late fusion of both SLIC and window features
    def __init__(self, opt):
        super(DeepGCN_slic_V3, self).__init__()
        channels = int(opt.n_filters)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        numRows = opt.numRows
        self.Stem_slic = Stem_slic(out_dim=channels, act=act, numRows=numRows)
        self.Stem_tiny = Stem_tiny(out_dim=channels, act=act)
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = (numRows*numRows) // max(num_knn)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        if opt.use_dilation:
            self.backbone1 = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone1 = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, originalInput):
        featMaps1, featMaps2 = [], []
        inputs = inputs.cuda()
        x_slic, x_slic_feat = self.Stem_slic(inputs, originalInput)   # x_slic: SLIC map of initial image, x: initialized features
        x_conv_feat = self.Stem_tiny(inputs) + self.pos_embed
        B, C, H, W = x_conv_feat.shape
        
        for i in range(self.n_blocks):
            x_slic_feat = self.backbone1[i](x_slic_feat)
            x_conv_feat = self.backbone2[i](x_conv_feat)
            featMaps1.append(x_slic_feat.detach().clone())
            featMaps2.append(x_conv_feat.detach().clone())
        
        x_slic_feat = F.adaptive_avg_pool2d(x_slic_feat, 1)
        x_conv_feat = F.adaptive_avg_pool2d(x_conv_feat, 1)
        
        x = x_slic_feat*0.5 + x_conv_feat*1.5
        return self.prediction(x).squeeze(-1).squeeze(-1), x_slic, [featMaps1, featMaps2]
        

class DeepGCN_baseline(torch.nn.Module):
    # V2: Samples square window features based on SLIC assignment
    # V3: Late fusion of both SLIC and window features
    def __init__(self, opt):
        super(DeepGCN_baseline, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        numRows = opt.numRows
        self.Stem_tiny = Stem_tiny(out_dim=channels, act=act)
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = (numRows*numRows) // max(num_knn)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        if opt.use_dilation:
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, originalInput):
        featMaps1, featMaps2 = [], []
        inputs = inputs.cuda()
        x_conv_feat = self.Stem_tiny(inputs) + self.pos_embed
        B, C, H, W = x_conv_feat.shape
        
        for i in range(self.n_blocks):
            x_conv_feat = self.backbone2[i](x_conv_feat)
            featMaps2.append(x_conv_feat.detach().clone())
        
        x_conv_feat = F.adaptive_avg_pool2d(x_conv_feat, 1)
        
        x = x_conv_feat
        return self.prediction(x).squeeze(-1).squeeze(-1), None, [featMaps1, featMaps2]

class DeepGCN_slic_V2_1(torch.nn.Module):
    # V2: Samples square window features based on SLIC assignment
    # V2_1: adjust number of SLIC clusters
    def __init__(self, opt):
        super(DeepGCN_slic_V2, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        numRows = opt.numRows
        # self.stem = Stem(out_dim=channels, act=act)
        # self.stem_tiny = Stem_tiny(out_dim=channels, act=act)
        self.Stem_slic = Stem_slic_V2_1(out_dim=channels, act=act, numRows=numRows)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = (numRows*numRows) // max(num_knn)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                            FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                            ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, originalInput):
        # ORIGINAL VERSION

        # featMaps = []
        # inputs = inputs.cuda()
        # x_slic, x, x_convFeat = self.Stem_slic(inputs, originalInput)   # x_slic: SLIC map of initial image, x: initialized features
        # x=x + x_convFeat
        # # pdb.set_trace()
        # B, C, H, W = x.shape
        
        # for i in range(self.n_blocks):
        #     x = self.backbone[i](x)
        #     featMaps.append(x.detach().clone())

        # x = F.adaptive_avg_pool2d(x, 1)
        # return self.prediction(x).squeeze(-1).squeeze(-1), x_slic, featMaps
        
        # CONV FUSION VERSION
        featMaps, featMaps_conv = [], []
        inputs = inputs.cuda()
        x_slic, x, x_convFeat = self.Stem_slic(inputs, originalInput)   # x_slic: SLIC map of initial image, x: initialized features

        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
            x_convFeat = self.backbone2[i](x_convFeat)
            featMaps.append(x.detach().clone())
            featMaps_conv.append(x_convFeat.detach().clone())

        x = F.adaptive_avg_pool2d(x, 1)
        x_convFeat = F.adaptive_avg_pool2d(x_convFeat, 1)
        # x = 0.5*x + 1.5*x_convFeat
        x=x_convFeat
        return self.prediction(x).squeeze(-1).squeeze(-1), x_slic, [featMaps, featMaps_conv]

class DeepGCN_slic_V2(torch.nn.Module):
    # V2: Samples square window features based on SLIC assignment
    def __init__(self, opt):
        super(DeepGCN_slic_V2, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        numRows = opt.numRows
        # self.stem = Stem(out_dim=channels, act=act)
        # self.stem_tiny = Stem_tiny(out_dim=channels, act=act)
        self.Stem_slic = Stem_slic_V2(out_dim=channels, act=act, numRows=numRows)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = (numRows*numRows) // max(num_knn)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                            FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                            ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
            self.backbone2 = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, originalInput):
        # ORIGINAL VERSION

        # featMaps = []
        # inputs = inputs.cuda()
        # x_slic, x, x_convFeat = self.Stem_slic(inputs, originalInput)   # x_slic: SLIC map of initial image, x: initialized features
        # x=x + x_convFeat
        # # pdb.set_trace()
        # B, C, H, W = x.shape
        
        # for i in range(self.n_blocks):
        #     x = self.backbone[i](x)
        #     featMaps.append(x.detach().clone())

        # x = F.adaptive_avg_pool2d(x, 1)
        # return self.prediction(x).squeeze(-1).squeeze(-1), x_slic, featMaps
        
        # CONV FUSION VERSION
        featMaps, featMaps_conv = [], []
        inputs = inputs.cuda()
        x_slic, x, x_convFeat = self.Stem_slic(inputs, originalInput)   # x_slic: SLIC map of initial image, x: initialized features

        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
            # x_convFeat = self.backbone2[i](x_convFeat)
            featMaps.append(x.detach().clone())
            featMaps_conv.append(x_convFeat.detach().clone())

        x = F.adaptive_avg_pool2d(x, 1)
        # x_convFeat = F.adaptive_avg_pool2d(x_convFeat, 1)
        # x = 0.5*x + 1.5*x_convFeat
        # x=x_convFeat
        return self.prediction(x).squeeze(-1).squeeze(-1), x_slic, [featMaps, featMaps_conv]
        
class DeepGCN_slic(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN_slic, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        numRows = opt.numRows
        # self.stem = Stem(out_dim=channels, act=act)
        # self.stem_tiny = Stem_tiny(out_dim=channels, act=act)
        self.Stem_slic = Stem_slic(out_dim=channels, act=act, numRows=numRows)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = (numRows*numRows) // max(num_knn)
        
        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, originalInput):
        featMaps = []
        x_slic, x = self.Stem_slic(inputs, originalInput)
        # pdb.set_trace()
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
            featMaps.append(x.detach().clone())

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1), x_slic, featMaps

class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        
        self.stem = Stem(out_dim=channels, act=act)
        self.stem_tiny = Stem_tiny(out_dim=channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem_tiny(inputs) + self.pos_embed
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

@register_model
def vig_ti_64_gelu_14by14_baseline(pretrained=False, **kwargs):         # 230525 edited
    class OptInit:
        def __init__(self, num_classes=200, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 8 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.numRows = 14
            
    opt = OptInit(**kwargs)
    model = DeepGCN_baseline(opt)
    model.default_cfg = default_cfgs['gnn_patch16_64']
    return model

@register_model
def vig_ti_64_gelu_14by14_slic_V3(pretrained=False, **kwargs):         # 230525 edited
    class OptInit:
        def __init__(self, num_classes=200, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 8 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.numRows = 14
            
    opt = OptInit(**kwargs)
    model = DeepGCN_slic_V3(opt)
    model.default_cfg = default_cfgs['gnn_patch16_64']
    return model

@register_model
def vig_ti_64_gelu_14by14_slic_V2(pretrained=False, **kwargs):         # 230525 edited
    class OptInit:
        def __init__(self, num_classes=200, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 8 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.numRows = 14
            
    opt = OptInit(**kwargs)
    model = DeepGCN_slic_V2(opt)
    model.default_cfg = default_cfgs['gnn_patch16_64']
    return model

@register_model
def vig_ti_64_gelu_14by14_slic(pretrained=False, **kwargs):         # 230525 edited
    class OptInit:
        def __init__(self, num_classes=200, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 8 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.numRows = 14
            
    opt = OptInit(**kwargs)
    model = DeepGCN_slic(opt)
    model.default_cfg = default_cfgs['gnn_patch16_64']
    return model

@register_model
def vig_ti_64_gelu_22by22_slic(pretrained=False, **kwargs):         # 230525 edited
    class OptInit:
        def __init__(self, num_classes=200, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 8 # number of basic blocks in the backbone
            self.n_filters = 96 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.numRows = 22

    opt = OptInit(**kwargs)
    model = DeepGCN_slic(opt)
    model.default_cfg = default_cfgs['gnn_patch16_64']
    return model
