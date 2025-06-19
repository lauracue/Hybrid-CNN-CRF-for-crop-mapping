"""
@author: Laura Elena Cué La Rosa

"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.CRFLayer import CRF
import numpy as np


def get_inplanes():
    return [64, 128, 256, 256]


def conv3x3x3(in_planes, out_planes, t_size = 5,
              t_pad=2, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(t_size,3,3),
                     stride=stride,
                     padding=(t_pad,1,1),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight,
                                  ignore_index = self.ignore_index) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, 
                 t_size = 5, t_pad=2, 
                 stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, t_size, t_pad, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, t_size, t_pad)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=5,
                 conv1_t_stride=1,
                 shortcut_type='B'):
        super().__init__()

        block_inplanes = [int(x * 1) for x in block_inplanes]

        self.in_planes = block_inplanes[0]

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(1, 1, 1),
                               padding=(conv1_t_size // 2, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type,
                                       t_size=conv1_t_size,
                                       t_pad=conv1_t_size // 2,
                                       stride=(1,2,2))
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       t_size=conv1_t_size,
                                       t_pad=conv1_t_size // 2,
                                       stride=(1,2,2))
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       t_size=conv1_t_size,
                                       t_pad=conv1_t_size // 2,
                                       stride=(1,2,2))
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       t_size=conv1_t_size,
                                       t_pad=conv1_t_size // 2,
                                       stride=(1,2,2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, t_size, t_pad, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  t_size=t_size, 
                  t_pad=t_pad,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, t_size, t_pad))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        return x, low_level_feat
    
    
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, 
                 t_size, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = (1,1,1)
            padding = (0,0,0)
        else:
            kernel_size = (t_size, 3, 3)
            padding = (t_size//2, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, 
                                            kernel_size=kernel_size,
                                            stride=1, 
                                            padding=padding, 
                                            dilation=(1,rate,rate), 
                                            bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

class DeepLabv3_plus(nn.Module):
    def __init__(self, model_depth, 
                 num_channels=3,
                 inp_seq = 14,
                 n_classes=21,
                 length = 9,
                 t_kernel_size = 5,
                 transmat = None,
                 w_cross = 1.0,
                 w_crf = 0.0,
                 batch_size = 8,
                 psize = 64,
                 tr_trans = False,
                 global_tr = False,
                 stop_grad = False,
                 softmax = False,
                 sigmoid = False,
                 tanh = False):
        super(DeepLabv3_plus, self).__init__()
        self.num_channels = num_channels
        self.inp_seq = inp_seq
        self.n_classes = n_classes
        self.length = length
        self.transmat = transmat
        self.w_cross = w_cross
        self.w_vit = w_crf
        self.tr_m = tr_trans
        self.global_tr = global_tr
        self.stop_grad = stop_grad
        self.t_kernel_size = t_kernel_size 
        self.batch = batch_size
        self.psize = psize
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.tanh = tanh
        
        self.zeromap =  torch.Tensor(np.zeros((self.batch,1,self.length,self.psize,self.psize))).float().cuda()
        
        self.crf = CRF(num_tags=self.n_classes, seq=self.length,
                       transmat=self.transmat,learn_ind= self.tr_m,
                       stop_grad = self.stop_grad)

        # Atrous Conv
        assert model_depth in [10, 18, 34]
    
        if model_depth == 10:
            self.resnet_features = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), 
                                          n_input_channels=self.num_channels,
                                          conv1_t_size=self.t_kernel_size,
                                          conv1_t_stride=1)
        elif model_depth == 18:
            self.resnet_features = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), 
                                          n_input_channels=self.num_channels,
                                          conv1_t_size=self.t_kernel_size,
                                          conv1_t_stride=1)
        elif model_depth == 34:
            self.resnet_features = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), 
                                          n_input_channels=self.num_channels,
                                          conv1_t_size=self.t_kernel_size,
                                          conv1_t_stride=1)

        # ASPP
        rates = [1, 3, 6, 9]

        self.aspp1 = ASPP_module(128, 128, self.t_kernel_size, rate=rates[0])
        self.aspp2 = ASPP_module(128, 128, self.t_kernel_size, rate=rates[1])
        self.aspp3 = ASPP_module(128, 128, self.t_kernel_size, rate=rates[2])
        self.aspp4 = ASPP_module(128, 128, self.t_kernel_size, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((self.length, 1, 1)),
                                             nn.Conv3d(128, 128, 
                                                       kernel_size=(self.t_kernel_size,1,1),
                                                       stride=1, 
                                                       padding=(self.t_kernel_size//2,0,0), 
                                                       bias=False),
                                             nn.BatchNorm3d(128),
                                             nn.ReLU())

        self.conv1 = nn.Conv3d(128*5, 128, 
                               kernel_size=(self.t_kernel_size,1,1),
                               stride=1, 
                               padding=(self.t_kernel_size//2,0,0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(128)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv3d(64, 128, 
                               kernel_size=(self.t_kernel_size,1,1),
                               stride=1, 
                               padding=(self.t_kernel_size//2,0,0), 
                               bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        
        self.p_conv = nn.Sequential(nn.Conv3d(256, 128, 
                                                 kernel_size=(self.t_kernel_size,3,3), 
                                                 stride=1, 
                                                 padding=(self.t_kernel_size//2,1,1), bias=False),
                                       nn.BatchNorm3d(128),
                                       nn.ReLU())

        self.last_conv = nn.Sequential(nn.Conv3d(128, 128, 
                                                 kernel_size=(self.t_kernel_size,3,3), 
                                                 stride=1, 
                                                 padding=(self.t_kernel_size//2,1,1), bias=False),
                                       nn.BatchNorm3d(128),
                                       nn.ReLU())
        
        self.drop = nn.Dropout3d(p=0.1)
        
        self.classifer = nn.Conv3d(128, self.n_classes, 
                                                 kernel_size=(self.t_kernel_size,1,1), 
                                                 stride=1, 
                                                 padding=(self.t_kernel_size//2,0,0), bias=False)
        
        self.masktemp = nn.Conv3d(self.inp_seq, self.length, 
                                   kernel_size=(1,3,3), 
                                   stride=1, 
                                   padding=(0,1,1), bias=False)

    def __build_features(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = nn.functional.interpolate(x, size=low_level_features.size()[2:], mode='trilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.p_conv(x)
        x = nn.functional.interpolate(x, size=input.size()[2:], mode='trilinear', align_corners=True)
        # x = self.last_conv(x)
        x = x.transpose(1,2)
        x = self.masktemp(x)
        x = x.transpose(2,1)
        x = self.drop(x)
        x = self.classifer(x)
        
        if self.softmax:
            x = F.softmax(x, dim=1)
        if self.sigmoid:
            x = torch.sigmoid(x)
        elif self.tanh:
            x = torch.tanh(x)
        
        return x
    
    def forward(self, x, indic = False, tags=None, only_tag = False, overlap=None):
        features = self.__build_features(x)
        features = features.transpose(1,2)
        if indic:
            if only_tag:
                features = features[:,:,:,overlap//2:self.psize-overlap//2,overlap//2:self.psize-overlap//2]
                tags = tags[:,:,overlap//2:self.psize-overlap//2,overlap//2:self.psize-overlap//2]
            crop_seq = self.crf(features,tags)
            if only_tag:
                features, tag = self._validate(features, tags)
                return features, crop_seq, tag
            else:
                return features, crop_seq
        else:
            return features
    
    def _validate(self, emissions, tags=None):
        if emissions.dim() !=5:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        
        try:
            B, L, C, H, W = emissions.shape
            emissions = emissions.permute(0,3,4,1,2).contiguous()
            emissions = emissions.view(B*H*W, L, C)
            
            tags = tags.permute(0,2,3,1).contiguous()
            tags = tags.view(B*H*W, L)
            
            emissions = emissions[tags[:,0]<C]
            tags = tags[tags[:,0]<C]
            return emissions, tags
        except:
            return emissions, 0            

    
    def loss(self, features, ys, epoch, start_warm=False):
        if self.w_vit > 0.0:
            loss_seq = self.crf.loss(features, ys.long())
        if self.w_cross>0.0:
            features, ys = self._validate(features, tags=ys)
            features = features.transpose(1,2)
            loss_corss = nn.CrossEntropyLoss()
            loss_class = loss_corss(features, ys.long())
        if start_warm:
            if epoch > start_warm:
                total_loss = self.w_vit*loss_seq + self.w_cross*loss_class
            else:
                return loss_class
        else:
            if self.w_vit > 0.0 and self.w_cross==0.0:
                return loss_seq
            elif self.w_vit > 0.0 and self.w_cross > 0.0:
                total_loss = self.w_vit*loss_seq + self.w_cross*loss_class
                return total_loss
            else:
                return loss_class
            

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()


if __name__ == "__main__":
    model = DeepLabv3_plus(10, 
                 num_channels=2,
                 inp_seq = 14,
                 n_classes=11,
                 length = 9,
                 t_kernel_size = 5,
                 transmat = None,
                 w_cross = 1.0,
                 w_crf = 0.0,
                 tr_trans = True,
                 global_tr = False,
                 stop_grad = False)
    model.eval()
    image = torch.randn(1, 2, 14, 128, 128)
    with torch.no_grad():
        output, crop_seq = model.forward(image)
    print(output.size())
    print(crop_seq.size())
    

    # crop_seq = torch.reshape(crop_seq, (B, L, H, W))

