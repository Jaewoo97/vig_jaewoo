# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import numpy as np
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from fast_slic import Slic
from PIL import Image
import pdb
import PIL
import matplotlib.pyplot as plt

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
        
    def slic_init(self, x, org_x):
        batchSize = len(x)
        # if isinstance(org_x, PIL.Image.Image):
        #     org_x = np.asarray(org_x)
        if torch.is_tensor(org_x):
            org_x = org_x.detach().cpu().numpy().astype('uint8')
        sliced = np.zeros((batchSize, x.shape[2], x.shape[3])).astype('int16')
        
        print('Processing slic on batch, batchsize: '+str(batchSize))
        
        if org_x.shape[-1] != 3:
            org_x=org_x.transpose(0,2,3,1)
        for batchIdx in range(batchSize):
            sliced[batchIdx] = self.slic.iterate(org_x[batchIdx].copy(order='C'))      # Iterate required input: H X W X 3

        print('Processing slic on batch / Done.')
        
        # TODO: return torch tensor with 10 x 10 x feature dimension
        x, sliced = torch.tensor(x).cuda(), torch.tensor(sliced).cuda()
        x_out = torch.zeros((batchSize,11,14,14))     # Dimension of initial SLIC map
        
        print('Creating x_out')
        
        for batchIdx in range(batchSize):
            for segIdx in range(self.numSegments):   
                if segIdx not in sliced[batchIdx]:
                    continue
                idcs = (sliced[batchIdx]==segIdx).nonzero()
                x_out[batchIdx, :2, int(segIdx/self.numRows), segIdx%self.numRows] = idcs.float().mean(0)
                rgbs = x[batchIdx, :, idcs[:,0], idcs[:,1]]
                x_out[batchIdx, 2:5, int(segIdx/self.numRows), segIdx%self.numRows] = rgbs.float().mean(1) 
                stds = rgbs.float().std(1)
                for idx, std in enumerate(stds):
                    stds[idx] = 0.0 if math.isnan(std) else std
                x_out[batchIdx, 5:8, int(segIdx/self.numRows), segIdx%self.numRows] = stds
                x_out[batchIdx, 8:11, int(segIdx/self.numRows), segIdx%self.numRows] = x[batchIdx, :, int(idcs.float().mean(0)[0]), int(idcs.float().mean(0)[1])]
                
        print('Creating x_out / Done.')
        
        x_out_loop = x_out.clone()
        x_out = torch.zeros((batchSize,11,14,14))     # Dimension of initial SLIC map
        batch_indices = torch.arange(batchSize).unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        seg_indices = torch.arange(self.numSegments).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()

        seg_mask = (sliced.unsqueeze(1) == seg_indices).float()

        idcs = (seg_mask.nonzero(as_tuple=False).squeeze()[:, 1:]).float().mean(0)
        pdb.set_trace()
        x_out[batch_indices, :2, seg_indices // self.numRows, seg_indices % self.numRows] = idcs

        rgbs = x[batch_indices, :, idcs[:, 0].long(), idcs[:, 1].long()]
        x_out[batch_indices, 2:5, seg_indices // self.numRows, seg_indices % self.numRows] = rgbs.mean(1)

        stds = rgbs.std(1)
        stds[torch.isnan(stds)] = 0.0
        x_out[batch_indices, 5:8, seg_indices // self.numRows, seg_indices % self.numRows] = stds

        mean_idcs = idcs.long()
        x_out[batch_indices, 8:11, seg_indices // self.numRows, seg_indices % self.numRows] = x[batch_indices, :, mean_idcs[:, 0], mean_idcs[:, 1]]
        
        print('Creating x_out 2 / Done.')
        pdb.set_trace()
        return sliced.clone(), x_out.cuda()

    def forward(self, x, org_x):
        # if x.dtype!=
        batchSize = x.shape[0]
        x_slic, x = self.slic_init(x, org_x)        # x_slic: B, 64, 64 / x: B, 11, 14, 14
        x = self.convs(x)
        return x_slic, x
    
    # def unNorm(self, img):
    #     img[:,0,:,:] = img[:,0,:,:]*DATA_STD[0] + DATA_MEAN[0]
    #     img[:,1,:,:] = img[:,1,:,:]*DATA_STD[1] + DATA_MEAN[1]
    #     img[:,2,:,:] = img[:,2,:,:]*DATA_STD[2] + DATA_MEAN[2]
    #     pdb.set_trace()
    #     img=img*255.0
    #     return img.int()
    
    # def Norm(self, img):
    #     img[:,0,:,:] = (img[:,0,:,:] - DATA_MEAN[0]) / DATA_STD[0] 
    #     img[:,1,:,:] = (img[:,1,:,:] - DATA_MEAN[1]) / DATA_STD[1] 
    #     img[:,2,:,:] = (img[:,2,:,:] - DATA_MEAN[2]) / DATA_STD[2] 
    #     return img

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
        # self.stem = Stem(out_dim=channels, act=act)
        # self.stem_tiny = Stem_tiny(out_dim=channels, act=act)
        self.Stem_slic = Stem_slic(out_dim=channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
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

    opt = OptInit(**kwargs)
    model = DeepGCN_slic(opt)
    model.default_cfg = default_cfgs['gnn_patch16_64']
    return model

@register_model
def vig_ti_64_gelu(pretrained=False, **kwargs):         # 230523 baseline
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
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

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_64']
    return model

@register_model
def vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 320 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 640 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model
