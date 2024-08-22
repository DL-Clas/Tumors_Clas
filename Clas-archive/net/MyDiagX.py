import torch
import torch.nn as nn
from skimage.segmentation import slic
import cv2
from skimage.measure import regionprops
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


class ResBlock_Separate(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock_Separate, self).__init__()
        self.inchannel, self.outchannel, self.stride = inchannel, outchannel, stride
        self.depth1 = nn.Conv2d(inchannel, inchannel, kernel_size = 3,
                                stride = stride, padding = 1, bias = False, groups = inchannel)
        self.point1 = nn.Conv2d(inchannel, outchannel, 1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace = True)
        self.depth2 = nn.Conv2d(outchannel, outchannel, kernel_size = 3,
                                stride = 1, padding = 1, bias = False, groups = outchannel)
        self.point2 = nn.Conv2d(outchannel, outchannel, 1)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.params_xy = nn.Parameter(torch.Tensor(1, inchannel, 7, 7), requires_grad = True)
        nn.init.ones_(self.params_xy)

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad = True)
        nn.init.ones_(self.params_zx)

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad = True)
        nn.init.ones_(self.params_zy)

        self.downsample = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.point1(self.depth1(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.point2(self.depth2(out))
        out = self.bn2(out)
        if self.stride != 1 or self.inchannel != self.outchannel:
            out = out + self.downsample(
                x * F.interpolate(self.params_zy, size = x.shape[2:4], mode = 'bilinear', align_corners = True)) + \
                  self.downsample(
                      x * F.interpolate(self.params_zx, size = x.shape[2:4], mode = 'bilinear', align_corners = True))
        else:
            out = out + self.downsample(
                x * F.interpolate(self.params_xy, size = x.shape[2:4], mode = 'bilinear', align_corners = True))
        out = self.relu(out)

        return out


class Bottleneck_Separate(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck_Separate, self).__init__()
        self.inchannel, self.outchannel, self.stride = inchannel, outchannel, stride
        self.depth1 = nn.Conv2d(inchannel, inchannel, kernel_size = 1, stride = stride, padding = 0, bias = False)
        self.point1 = nn.Conv2d(inchannel, int(outchannel / 4), 1)
        self.bn1 = nn.BatchNorm2d(int(outchannel / 4))
        self.relu1 = nn.ReLU(inplace = True)

        self.depth2 = nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.point2 = nn.Conv2d(int(outchannel / 4), int(outchannel / 4), 1)
        self.bn2 = nn.BatchNorm2d(int(outchannel / 4))
        self.relu2 = nn.ReLU(inplace = True)

        self.depth3 = nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.point3 = nn.Conv2d(int(outchannel / 4), outchannel, 1)
        self.bn3 = nn.BatchNorm2d(outchannel)

        self.params_xy = nn.Parameter(torch.Tensor(1, inchannel, 7, 7), requires_grad = True)
        nn.init.ones_(self.params_xy)

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad = True)
        nn.init.ones_(self.params_zx)

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, inchannel, 7), requires_grad = True)
        nn.init.ones_(self.params_zy)

        self.downsample = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = stride, bias = False),
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
                x * F.interpolate(self.params_zy, size = x.shape[2:4], mode = 'bilinear', align_corners = True)) + \
                  self.downsample(
                      x * F.interpolate(self.params_zx, size = x.shape[2:4], mode = 'bilinear', align_corners = True))
        else:
            out = out + self.downsample(
                x * F.interpolate(self.params_xy, size = x.shape[2:4], mode = 'bilinear', align_corners = True))
        out = F.relu(out)

        return out


class Conv_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_1, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size = 1)

    def forward(self, x):
        # 前向传播，将输入通过1x1卷积层
        x = self.conv1x1(x)
        return x


class Superpixel_calculate():
    def __init__(self, image, n_segments):
        image = image.detach().numpy()
        self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.n_segments = n_segments
        self.segments = slic(self.image, n_segments = self.n_segments,
                             compactness = 100, sigma = 1, start_label = 1)

    def Adjacency_matrix(self):
        # 初始化邻接矩阵
        super_labels = self.segments
        num_super = np.max(super_labels) + 1
        adjacency_matrix = np.zeros((num_super, num_super), dtype = int)
        # 填充邻接矩阵
        for i in range(self.image.shape[0] - 1):
            for j in range(self.image.shape[1] - 1):
                current_label = super_labels[i, j]
                right_label = super_labels[i, j + 1]
                bottom_label = super_labels[i + 1, j]

                if current_label != right_label:
                    adjacency_matrix[current_label, right_label] = 1
                    adjacency_matrix[right_label, current_label] = 1

                if current_label != bottom_label:
                    adjacency_matrix[current_label, bottom_label] = 1
                    adjacency_matrix[bottom_label, current_label] = 1

        src, dst = np.where(adjacency_matrix == 1)
        src = torch.tensor(src - 1)
        dst = torch.tensor(dst - 1)
        return src, dst

    def Superpixel_features(self):
        # 计算每个超像素块的特征
        super_fea = []
        region_props = regionprops(self.segments, intensity_image = self.image)
        for idx, prop in enumerate(region_props):
            # 获取中心坐标
            center_x = prop.centroid[0]
            center_y = prop.centroid[1]
            # 获取最大灰度值
            mean_intensity = prop.mean_intensity.max()
            # 获取总像素点数
            area = prop.area
            super_fea.append([center_x, center_y, mean_intensity, area])

        # 标准化处理
        super_fea = torch.tensor(super_fea, dtype = torch.float32)
        subset_data = super_fea[:, 2:]
        mean = subset_data.mean(dim = 0)
        std = subset_data.std(dim = 0)
        normal_data = (subset_data - mean) / (std + 1)
        super_fea[:, 2:] = normal_data

        return super_fea


class GCN(nn.Module):
    # 定义GCN模型
    def __init__(self, in_feats, nodes, hidden_size, output_size):
        super(GCN, self).__init__()
        self.nodes = nodes
        self.output_size = output_size

        self.gconv1 = GraphConv(in_feats, hidden_size)
        self.gconv2 = GraphConv(hidden_size, output_size)  # output = [nodes, output_size]
        self.fc = nn.Linear(nodes, 1)

    def forward(self, g, inputs):
        g1 = torch.relu(self.gconv1(g, inputs))
        g2 = torch.relu(self.gconv2(g, g1))
        out = self.fc(g2.t())

        return out


def Channel_Multi(G_fea, B_fea):
    # 对应通道融合
    G_fea = G_fea.view(G_fea.size(1), G_fea.size(0), 1, 1)
    G_fea = G_fea.expand_as(B_fea)
    output = G_fea * B_fea

    return output


def make_graph(l_conv1, B_for, B_fon, Sp, gcn):
    L = l_conv1(B_for)  # 层输出通道处理
    S = Superpixel_calculate(L[0].squeeze(), Sp)  # 特征图超像素计算
    M = dgl.graph(S.Adjacency_matrix())  # 构建图结构
    G = gcn(M, S.Superpixel_features())  # GCN
    for i in range(1, L.size(0)):
        S = Superpixel_calculate(L[i].squeeze(), Sp)  # 特征图超像素计算
        M = dgl.graph(S.Adjacency_matrix())  # 构建图结构
        G2 = gcn(M, S.Superpixel_features())  # GCN
        G = torch.cat([G, G2], dim = 1)
    # 对应通道融合
    G = G.view(G.size(1), G.size(0), 1, 1)
    G = G.expand_as(B_fon)
    F = G * B_fon

    return F


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

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
            u = x.mean(1, keepdim = True)
            s = (x - u).pow(2).mean(1, keepdim = True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Group_Norm(nn.Module):
    def __init__(self, cl, k_size=3, pad=1):
        super(Group_Norm, self).__init__()
        self.gn = nn.Sequential(
            LayerNorm(normalized_shape = cl, data_format = 'channels_first'),
            nn.Conv2d(cl, cl, kernel_size = 3, stride = 1,
                      padding = (k_size + (k_size - 1) * (pad - 1)) // 2,
                      dilation = pad, groups = cl)
        )

    def forward(self, x):
        # 前向传播，将输入通过1x1卷积层
        x = self.gn(x)
        return x


class Group_Feature_Fusion(nn.Module):
    def __init__(self, fea_num, al, zl, sl):
        super().__init__()
        k_size = 3
        d_list = [1, 2, 5, 7]

        self.avgpool_layer = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.gn_layers = nn.ModuleDict()
        for lay in range(0, fea_num):
            gn_layer = Group_Norm(al, k_size, d_list[lay])
            self.gn_layers[f'gnv{lay}'] = gn_layer
        self.gn_layers[f'gnv{fea_num - 1}'] = Group_Norm(zl, k_size, d_list[fea_num - 1])

        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape = al * (fea_num - 1) + zl, data_format = 'channels_first'),
            nn.Conv2d(al * (fea_num - 1) + zl, sl, 1)
        )

    def forward(self, **kwargs):
        feature = kwargs
        nums = len(feature)
        keys = feature.keys()
        Group_fea = []
        # 统一特征图尺寸
        for i, x in enumerate(feature):
            for j in range(nums - i - 1):
                feature[x] = self.avgpool_layer(feature[x])
        # 按组划分通道
        for key, value in feature.items():
            feature[key] = torch.chunk(feature[key], nums, dim = 1)
        # 按组拼接
        for group_i in range(nums):
            Grop_fea_tmp = torch.tensor([])
            for key in keys:
                if Grop_fea_tmp.numel() == 0:
                    Grop_fea_tmp = feature[key][group_i]
                else:
                    Grop_fea_tmp = torch.cat((Grop_fea_tmp, feature[key][group_i]), dim = 1)
            Group_fea.append(Grop_fea_tmp)

        # 每组按不同d进行特征融合
        for name, gn in self.gn_layers.items():
            num = int(name.split('gnv')[1])
            Group_fea[num] = gn(Group_fea[num])

        # 将各组特征进行拼接
        Grop_fea_fu = torch.tensor([])
        for gp in range(0, len(Group_fea)):
            Grop_fea_fu = torch.cat((Grop_fea_fu, Group_fea[gp]), dim = 1)

        x = self.tail_conv(Grop_fea_fu)

        return x

# default argument follows 18 layers
class MyDiag_Model(nn.Module):
    def __init__(self, num_classes = 4, clist = [0, 0, 96, 151, 240], olist = [0, 0, 96, 146, 240],
                 slist = [0, 64, 128, 256, 512], nlist = [0, 2, 2, 2, 2], kind = ResBlock_Separate, num = 512) -> None:
        super(MyDiag_Model, self).__init__()
        self.num_classes, self.clist, self.olist, self.slist, self.nlist, self.num = num_classes, clist, olist, slist, nlist, num
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1, ceil_mode = False)

        self.layer1 = self.make_layer(kind, self.slist[1], self.nlist[1], stride = 1)
        self.layer2 = self.make_layer(kind, self.slist[2], self.nlist[2], stride = 2)
        self.layer3 = self.make_layer(kind, self.slist[3], self.nlist[3], stride = 2)
        self.layer4 = self.make_layer(kind, self.slist[4], self.nlist[4], stride = 2)

        self.l1_conv1 = Conv_1(self.slist[1], 1)
        self.gcn1 = GCN(4, 16, 32, self.slist[2])
        self.l2_conv1 = Conv_1(self.slist[2], 1)
        self.gcn2 = GCN(4, 9, 32, self.slist[3])
        self.l3_conv1 = Conv_1(self.slist[3], 1)
        self.gcn3 = GCN(4, 4, 32, self.slist[4])

        self.GFF2 = Group_Feature_Fusion(2, self.clist[2], self.olist[2], self.slist[2])
        self.GFF3 = Group_Feature_Fusion(3, self.clist[3], self.olist[3], self.slist[3])
        self.GFF4 = Group_Feature_Fusion(4, self.clist[4], self.olist[4], self.slist[4])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(self.num, self.num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        O1 = self.relu(self.bn1(self.conv1(x)))
        O2 = self.maxpool(O1)
        B1 = self.layer1(O2)

        B2 = self.layer2(B1)
        B2 = self.GFF2(x1 = B1, x2 = B2)
        F2 = make_graph(self.l1_conv1, B1, B2, 16, self.gcn1)

        B3 = self.layer3(B2)
        B3 = self.GFF3(x1 = B1, x2 = B2, x3 = B3)
        F3 = make_graph(self.l2_conv1, F2, B3, 9, self.gcn2)

        B4 = self.layer4(B3)
        B4 = self.GFF4(x1 = B1, x2 = B2, x3 = B3, x4 = B4)
        F4 = make_graph(self.l3_conv1, F3, B4, 4, self.gcn3)

        O3 = self.avgpool(F4 + B4)
        O4 = O3.view(O3.size(0), -1)
        out = self.fc(O4)
        return out


def MyDiag53(num_classes):
    model = MyDiag_Model(num_classes = num_classes, clist = [0, 0, 384, 599, 960], olist = [0, 0, 384, 594, 960],
                 slist = [0, 256, 512, 1024, 2048], nlist = [0, 3, 4, 6, 3], kind = Bottleneck_Separate, num = 4 * 512)

    return model


def MyDiag37(num_classes):
    model = MyDiag_Model(num_classes = num_classes, clist = [0, 0, 96, 151, 240], olist = [0, 0, 96, 146, 240],
        slist = [0, 64, 128, 256, 512], nlist = [0, 3, 4, 6, 3], kind = ResBlock_Separate, num = 512)

    return model

def MyDiag21(num_classes):
    model = MyDiag_Model(num_classes = num_classes, clist = [0, 0, 96, 151, 240], olist = [0, 0, 96, 146, 240],
        slist = [0, 64, 128, 256, 512], nlist = [0, 2, 2, 2, 2], kind = ResBlock_Separate, num = 512)

    return model

def MyDiag53_tiny(num_classes):
    model = MyDiag_Model(num_classes = num_classes, clist = [0, 0, 192, 300, 480], olist = [0, 0, 192, 296, 480],
                 slist = [0, 128, 256, 512, 1024], nlist = [0, 3, 4, 6, 3], kind = Bottleneck_Separate, num = 4 * 256)

    return model


def MyDiag37_tiny(num_classes):
    model = MyDiag_Model(num_classes = num_classes, clist = [0, 0, 48, 76, 120], olist = [0, 0, 48, 72, 120],
        slist = [0, 32, 64, 128, 256], nlist = [0, 3, 4, 6, 3], kind = ResBlock_Separate, num = 256)

    return model

def MyDiag21_tiny(num_classes):
    model = MyDiag_Model(num_classes = num_classes, clist = [0, 0, 48, 76, 120], olist = [0, 0, 48, 72, 120],
        slist = [0, 32, 64, 128, 256], nlist = [0, 2, 2, 2, 2], kind = ResBlock_Separate, num = 256)

    return model

if __name__ == '__main__':
    model = MyDiag53_tiny(5)
    x = torch.randn(8, 3, 256, 256)
    print("shape: ", model(x).shape)
