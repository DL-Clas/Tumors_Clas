import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import dgl
from dgl.nn.pytorch import GraphConv
from skimage.segmentation import slic
from skimage.measure import regionprops
import warnings

# ==========================================
# 1. ER-3DA Module: Efficient Residual blocks fused with 3-dimensional tensor attention
# ==========================================
class ER_3DA_ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ER_3DA_ResBlock, self).__init__()
        self.inchannel, self.outchannel, self.stride = inchannel, outchannel, stride
        self.depth1 = nn.Conv2d(inchannel, inchannel, kernel_size=3,
                                stride=stride, padding=1, bias=False, groups=inchannel)
        self.point1 = nn.Conv2d(inchannel, outchannel, 1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        
        self.depth2 = nn.Conv2d(outchannel, outchannel, kernel_size=3,
                                stride=1, padding=1, bias=False, groups=outchannel)
        self.point2 = nn.Conv2d(outchannel, outchannel, 1)
        self.bn2 = nn.BatchNorm2d(outchannel)

        # 3D Tensor Attention Parameters (Corresponding to ER-3DA)
        self.params_xy = nn.Parameter(torch.Tensor(1, inchannel, 7, 7), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.params_zx = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.params_zy = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad=True)
        nn.init.ones_(self.params_zy)

        self.downsample = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.point1(self.depth1(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.point2(self.depth2(out))
        out = self.bn2(out)
        
        # Applying 3D Tensor Attention logic
        if self.stride != 1 or self.inchannel != self.outchannel:
            out = out + self.downsample(
                x * F.interpolate(self.params_zy, size=x.shape[2:4], mode='bilinear', align_corners=True)) + \
                  self.downsample(
                      x * F.interpolate(self.params_zx, size=x.shape[2:4], mode='bilinear', align_corners=True))
        else:
            out = out + self.downsample(
                x * F.interpolate(self.params_xy, size=x.shape[2:4], mode='bilinear', align_corners=True))
        out = self.relu(out)
        return out


class ER_3DA_Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ER_3DA_Bottleneck, self).__init__()
        self.inchannel, self.outchannel, self.stride = inchannel, outchannel, stride
        
        self.depth1 = nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=stride, padding=0, bias=False)
        self.point1 = nn.Conv2d(inchannel, int(outchannel / 4), 1)
        self.bn1 = nn.BatchNorm2d(int(outchannel / 4))
        self.relu1 = nn.ReLU(inplace=True)

        self.depth2 = nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size=3, stride=1, padding=1, bias=False)
        self.point2 = nn.Conv2d(int(outchannel / 4), int(outchannel / 4), 1)
        self.bn2 = nn.BatchNorm2d(int(outchannel / 4))
        self.relu2 = nn.ReLU(inplace=True)

        self.depth3 = nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size=1, stride=1, padding=0, bias=False)
        self.point3 = nn.Conv2d(int(outchannel / 4), outchannel, 1)
        self.bn3 = nn.BatchNorm2d(outchannel)

        self.params_xy = nn.Parameter(torch.Tensor(1, inchannel, 7, 7), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.params_zx = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.params_zy = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad=True)
        nn.init.ones_(self.params_zy)

        self.downsample = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.point1(self.depth1(x))
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.point2(self.depth2(out))
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.point3(self.depth3(out))
        out = self.bn3(out)

        if self.stride != 1 or self.inchannel != self.outchannel:
            out = out + self.downsample(
                x * F.interpolate(self.params_zy, size=x.shape[2:4], mode='bilinear', align_corners=True)) + \
                  self.downsample(
                      x * F.interpolate(self.params_zx, size=x.shape[2:4], mode='bilinear', align_corners=True))
        else:
            out = out + self.downsample(
                x * F.interpolate(self.params_xy, size=x.shape[2:4], mode='bilinear', align_corners=True))
        out = F.relu(out)
        return out


class Conv_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)

# ==========================================
# 2. MFS-GD Module: Multi-scale Feature Fusion Strategy based on Group DenseNet
# ==========================================
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf) """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Group_Norm(nn.Module):
    def __init__(self, cl, k_size=3, pad=1):
        super(Group_Norm, self).__init__()
        self.gn = nn.Sequential(
            LayerNorm(normalized_shape=cl, data_format='channels_first'),
            nn.Conv2d(cl, cl, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (pad - 1)) // 2,
                      dilation=pad, groups=cl)
        )

    def forward(self, x):
        return self.gn(x)

class MFS_GD(nn.Module):
    """ Renamed from Group_Feature_Fusion to match manuscript """
    def __init__(self, fea_num, al, zl, sl):
        super().__init__()
        k_size = 3
        d_list = [1, 2, 5, 7]

        self.avgpool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gn_layers = nn.ModuleDict()
        for lay in range(0, fea_num):
            gn_layer = Group_Norm(al, k_size, d_list[lay])
            self.gn_layers[f'gnv{lay}'] = gn_layer
        self.gn_layers[f'gnv{fea_num - 1}'] = Group_Norm(zl, k_size, d_list[fea_num - 1])

        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=al * (fea_num - 1) + zl, data_format='channels_first'),
            nn.Conv2d(al * (fea_num - 1) + zl, sl, 1)
        )

    def forward(self, **kwargs):
        feature = kwargs
        nums = len(feature)
        keys = feature.keys()
        Group_fea = []
        
        for i, x in enumerate(feature):
            for j in range(nums - i - 1):
                feature[x] = self.avgpool_layer(feature[x])
                
        for key, value in feature.items():
            feature[key] = torch.chunk(feature[key], nums, dim=1)
            
        for group_i in range(nums):
            Grop_fea_tmp = torch.tensor([]).to(feature[list(keys)[0]][0].device)
            for key in keys:
                if Grop_fea_tmp.numel() == 0:
                    Grop_fea_tmp = feature[key][group_i]
                else:
                    Grop_fea_tmp = torch.cat((Grop_fea_tmp, feature[key][group_i]), dim=1)
            Group_fea.append(Grop_fea_tmp)

        for name, gn in self.gn_layers.items():
            num = int(name.split('gnv')[1])
            Group_fea[num] = gn(Group_fea[num])

        Grop_fea_fu = torch.tensor([]).to(Group_fea[0].device)
        for gp in range(0, len(Group_fea)):
            Grop_fea_fu = torch.cat((Grop_fea_fu, Group_fea[gp]), dim=1)

        x = self.tail_conv(Grop_fea_fu)
        return x


# ==========================================
# 3. GCF-2S Module: Graph Convolutional Feature extraction based on Superpixel Segmentation
# ==========================================
class Superpixel_calculate():
    def __init__(self, image, n_segments):
        # Detach and move to CPU safely for SLIC processing
        image = image.detach().cpu().numpy()
        self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.n_segments = n_segments
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.segments = slic(self.image, n_segments=self.n_segments,
                                 compactness=100, sigma=1, start_label=1)
            
        # Ensure consecutive mapping preventing out-of-bound errors
        self.unique_labels = np.unique(self.segments)
        label_map = {old: new for new, old in enumerate(self.unique_labels)}
        self.mapped_segments = np.vectorize(label_map.get)(self.segments)

    def Adjacency_matrix(self):
        num_super = len(self.unique_labels)
        adjacency_matrix = np.zeros((num_super, num_super), dtype=int)
        
        for i in range(self.image.shape[0] - 1):
            for j in range(self.image.shape[1] - 1):
                current_label = self.mapped_segments[i, j]
                right_label = self.mapped_segments[i, j + 1]
                bottom_label = self.mapped_segments[i + 1, j]

                if current_label != right_label:
                    adjacency_matrix[current_label, right_label] = 1
                    adjacency_matrix[right_label, current_label] = 1

                if current_label != bottom_label:
                    adjacency_matrix[current_label, bottom_label] = 1
                    adjacency_matrix[bottom_label, current_label] = 1

        src, dst = np.where(adjacency_matrix == 1)
        return torch.tensor(src), torch.tensor(dst)

    def Superpixel_features(self):
        super_fea = []
        region_props = regionprops(self.segments, intensity_image=self.image)
        for prop in region_props:
            center_x, center_y = prop.centroid[0], prop.centroid[1]
            mean_intensity = prop.mean_intensity.max()
            area = prop.area
            super_fea.append([center_x, center_y, mean_intensity, area])

        super_fea = torch.tensor(super_fea, dtype=torch.float32)
        subset_data = super_fea[:, 2:]
        
        if subset_data.size(0) > 1:
            mean = subset_data.mean(dim=0)
            std = subset_data.std(dim=0) + 1e-6
            normal_data = (subset_data - mean) / std
        else:
            normal_data = torch.zeros_like(subset_data)
            
        super_fea[:, 2:] = normal_data
        return super_fea


class GCN(nn.Module):
    def __init__(self, in_feats, nodes, hidden_size, output_size):
        super(GCN, self).__init__()
        self.nodes = nodes
        self.output_size = output_size

        self.gconv1 = GraphConv(in_feats, hidden_size)
        self.gconv2 = GraphConv(hidden_size, output_size)
        self.fc = nn.Linear(nodes, 1)

    def forward(self, g, inputs):
        g1 = torch.relu(self.gconv1(g, inputs))
        g2 = torch.relu(self.gconv2(g, g1))
        
        g2_t = g2.t()  # [output_size, num_nodes]
        
        # Dynamic Node Padding to prevent crashing and maintain pre-trained compatability
        current_nodes = g2_t.size(1)
        if current_nodes < self.nodes:
            pad = torch.zeros(g2_t.size(0), self.nodes - current_nodes, device=g2_t.device)
            g2_t = torch.cat([g2_t, pad], dim=1)
        elif current_nodes > self.nodes:
            g2_t = g2_t[:, :self.nodes]
            
        out = self.fc(g2_t) # [output_size, 1]
        return out


def GCF_2S(l_conv1, B_for, B_fon, Sp, gcn):
    """ Wrapper for GCF-2S Process logic replacing make_graph """
    device = B_for.device
    L = l_conv1(B_for)  # [B, 1, H, W]
    
    G_list = []
    # Safe batch loop iteration
    for i in range(L.size(0)):
        S = Superpixel_calculate(L[i].squeeze(0).cpu(), Sp)
        src, dst = S.Adjacency_matrix()
        
        # Mapping CPU structural matrices safely to GPU device
        M = dgl.graph((src, dst)).to(device)
        features = S.Superpixel_features().to(device)
        
        G_out = gcn(M, features) # [output_size, 1]
        G_list.append(G_out)
        
    G = torch.cat(G_list, dim=1) # [output_size, B]
    G = G.transpose(0, 1).contiguous().view(-1, B_fon.size(1), 1, 1) # [B, output_size, 1, 1]
    
    F = G * B_fon
    return F


# ==========================================
# 4. BTNet-TS Final Implementations
# ==========================================
class MyDiag_Model(nn.Module):
    def __init__(self, num_classes=4, clist=[0, 0, 96, 151, 240], olist=[0, 0, 96, 146, 240],
                 slist=[0, 64, 128, 256, 512], nlist=[0, 2, 2, 2, 2], kind=ER_3DA_ResBlock, num=512) -> None:
        super(MyDiag_Model, self).__init__()
        self.num_classes, self.clist, self.olist, self.slist, self.nlist, self.num = num_classes, clist, olist, slist, nlist, num
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.layer1 = self.make_layer(kind, self.slist[1], self.nlist[1], stride=1)
        self.layer2 = self.make_layer(kind, self.slist[2], self.nlist[2], stride=2)
        self.layer3 = self.make_layer(kind, self.slist[3], self.nlist[3], stride=2)
        self.layer4 = self.make_layer(kind, self.slist[4], self.nlist[4], stride=2)

        self.l1_conv1 = Conv_1(self.slist[1], 1)
        self.gcn1 = GCN(4, 16, 32, self.slist[2])
        self.l2_conv1 = Conv_1(self.slist[2], 1)
        self.gcn2 = GCN(4, 9, 32, self.slist[3])
        self.l3_conv1 = Conv_1(self.slist[3], 1)
        self.gcn3 = GCN(4, 4, 32, self.slist[4])

        self.GFF2 = MFS_GD(2, self.clist[2], self.olist[2], self.slist[2])
        self.GFF3 = MFS_GD(3, self.clist[3], self.olist[3], self.slist[3])
        self.GFF4 = MFS_GD(4, self.clist[4], self.olist[4], self.slist[4])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(self.num, self.num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inchannel, channels, s))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        O1 = self.relu(self.bn1(self.conv1(x)))
        O2 = self.maxpool(O1)
        
        B1 = self.layer1(O2)

        B2 = self.layer2(B1)
        B2 = self.GFF2(x1=B1, x2=B2)
        F2 = GCF_2S(self.l1_conv1, B1, B2, 16, self.gcn1)

        B3 = self.layer3(B2)
        B3 = self.GFF3(x1=B1, x2=B2, x3=B3)
        F3 = GCF_2S(self.l2_conv1, F2, B3, 9, self.gcn2)

        B4 = self.layer4(B3)
        B4 = self.GFF4(x1=B1, x2=B2, x3=B3, x4=B4)
        F4 = GCF_2S(self.l3_conv1, F3, B4, 4, self.gcn3)

        O3 = self.avgpool(F4 + B4)
        O4 = O3.view(O3.size(0), -1)
        out = self.fc(O4)
        return out


# ==========================================
# 5. Exportable Architecture Implementations
# ==========================================
def MyDiag53(num_classes):
    model = MyDiag_Model(num_classes=num_classes, clist=[0, 0, 384, 599, 960], olist=[0, 0, 384, 594, 960],
                 slist=[0, 256, 512, 1024, 2048], nlist=[0, 3, 4, 6, 3], kind=ER_3DA_Bottleneck, num=4 * 512)
    return model

def MyDiag37(num_classes):
    model = MyDiag_Model(num_classes=num_classes, clist=[0, 0, 96, 151, 240], olist=[0, 0, 96, 146, 240],
        slist=[0, 64, 128, 256, 512], nlist=[0, 3, 4, 6, 3], kind=ER_3DA_ResBlock, num=512)
    return model

def MyDiag21(num_classes):
    model = MyDiag_Model(num_classes=num_classes, clist=[0, 0, 96, 151, 240], olist=[0, 0, 96, 146, 240],
        slist=[0, 64, 128, 256, 512], nlist=[0, 2, 2, 2, 2], kind=ER_3DA_ResBlock, num=512)
    return model

def MyDiag53_tiny(num_classes):
    model = MyDiag_Model(num_classes=num_classes, clist=[0, 0, 192, 300, 480], olist=[0, 0, 192, 296, 480],
                 slist=[0, 128, 256, 512, 1024], nlist=[0, 3, 4, 6, 3], kind=ER_3DA_Bottleneck, num=4 * 256)
    return model

def MyDiag37_tiny(num_classes):
    model = MyDiag_Model(num_classes=num_classes, clist=[0, 0, 48, 76, 120], olist=[0, 0, 48, 72, 120],
        slist=[0, 32, 64, 128, 256], nlist=[0, 3, 4, 6, 3], kind=ER_3DA_ResBlock, num=256)
    return model

def MyDiag21_tiny(num_classes):
    model = MyDiag_Model(num_classes=num_classes, clist=[0, 0, 48, 76, 120], olist=[0, 0, 48, 72, 120],
        slist=[0, 32, 64, 128, 256], nlist=[0, 2, 2, 2, 2], kind=ER_3DA_ResBlock, num=256)
    return model

if __name__ == '__main__':
    # Test Block to ensure batching and tensor flow operates correctly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyDiag21_tiny(4).to(device)
    # Using batch_size=4 to ensure our GCF_2S iteration bug fix works flawlessly
    x = torch.randn(4, 3, 224, 224).to(device)
    print(f"Executing Forward Pass...")
    out = model(x)
    print(f"Success! Output Shape: {out.shape}")
