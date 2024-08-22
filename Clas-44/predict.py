import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from net.MyDiagX import MyDiag21 as create_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    image_path = "data_set/Normal_T1.jpg"
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)   # [N, C H, W]
    img = torch.unsqueeze(img, dim=0)   # 维度扩展
    json_path = "./calss_indices.json"
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    model = create_model(num_classes=44).to(device)   # GPU
    weights_path = "weights/MyDialog21.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()    # 关闭 Dorpout
    with torch.no_grad():
        output = torch.squeeze(model(img))      # 维度压缩
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        plt.title(print_res)
        plt.show()

if __name__ == '__main__':
    main()
