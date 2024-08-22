import torch
import time

# from net.transnext import transnext_small as create_model
# from net.rdnet import RDNet as create_model
# from net.shufflenetv1 import shufflenet_g2 as create_model
from net.resnet import ResNet18 as create_model
# from net.mobilenetv2 import MobileNetV2 as create_model
# from net.model import swin_tiny_patch4_window7_224 as create_model
# from net.MyDiagX import MyDiag53_tiny as create_model
# from net.DensNet import DenseNet121 as create_model

from calflops import calculate_flops

model = create_model(num_classes=5)
flops, macs, params = calculate_flops(model, input_shape=(1, 3, 224, 224))
print(flops, macs, params)


# def calculate_fps(model, input_image):
#     # 测量模型处理一帧图像所花费的时间
#     start_time = time.time()
#     net = model(1, 1)
#     # 假设模型接受的输入是图像 input_image，进行模型推理
#     output = net(input_image)
#     end_time = time.time()
#
#     # 计算处理一帧图像所花费的时间
#     processing_time = end_time - start_time
#
#     # 计算帧率
#     fps = 1 / processing_time
#
#     return fps
#
#
# # 假设 model 是你的分割模型，input_image 是输入的图像
# input_image = torch.randn(1, 1, 128, 128)
# fps = calculate_fps(create_model, input_image)
# print("FPS:", fps)
