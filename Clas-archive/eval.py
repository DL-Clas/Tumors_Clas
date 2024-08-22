import os
import argparse
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pylab as plt

# from net.transnext import transnext_small as create_model
# from net.rdnet import RDNet as create_model
# from net.shufflenetv1 import shufflenet_g2 as create_model
# from net.resnet import ResNet18 as create_model
# from net.mobilenetv2 import MobileNetV2 as create_model
# from net.model import swin_tiny_patch4_window7_224 as create_model
from net.MyDiagX import MyDiag21 as create_model
# from net.DensNet import DenseNet121 as create_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设定图像转换和数据加载
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_val = args.val_data  # 验证集数据文件夹的路径
    validation_dataset = datasets.ImageFolder(root=root_val, transform=data_transform)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,  # 验证数据集
        batch_size=args.batch_size,  # 批大小，每个批次包含20个样本
        shuffle=False  # 不对验证数据进行打乱，保持原有顺序
    )

    # model = vgg(model_name="vgg16", num_classes=4, init_weights=True)  # 实例化网络(5分类)
    model = create_model(num_classes=args.num_classes)

    # load model weights
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 初始化混淆矩阵和真实标签列表
    y_true = []
    y_pred = []

    # 逐批次进行预测
    with torch.no_grad():
        with tqdm(total=len(validation_loader), postfix=dict, mininterval=0.3) as pbar:
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # 将真实标签和预测标签添加到列表中
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())
                pbar.update(1)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    plt.matshow(conf_matrix, cmap=plt.cm.Blues)  # 根据最下面的图按自己需求更改颜色
    plt.colorbar()

    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            plt.annotate(conf_matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
            # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    plt.savefig('results/result.png')
    plt.show()

    # 计算准确性
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # 计算精准度
    precision = precision_score(y_true, y_pred, average='weighted')

    # 计算召回率
    recall = recall_score(y_true, y_pred, average='weighted')

    # 计算F1分数
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_data', type=str, default="data_set/archive_data/val")
    parser.add_argument('--weights', type=str, default='weights/MyDialog21.pth')
    opt = parser.parse_args()
    main(opt)