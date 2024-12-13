import torch
from PIL import Image
from torchvision import transforms

from Attention_UNet import AttU_Net

# 创建模型对象，并设置 deep supervision=True
model = AttU_Net(img_ch=3, output_ch=1, deepsupervision=True)

# 载入模型权重参数（可选）
model.load_state_dict(torch.load('saved_model/Attention_UNet_1_Prostate_2.pth'))

# 加载并预处理输入图像
input_image = Image.open('ProstateData/test/imagesTr/008.png')
input_image = transforms.ToTensor()(input_image)
input_image = input_image.unsqueeze(0)

# 在模型中执行前向传播，并在每个 Conv 和 Attention block 层级上存储特征
with torch.no_grad():
    model.eval()
    features = model(input_image)
