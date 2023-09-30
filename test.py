from PIL import Image
from torchvision import transforms
from utils import get_pixels
import torch
from model import ImageMLP
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    image_path= input("Enter image path: ")
    num_epochs=200

    transform = transforms.ToTensor()
    image= transform(Image.open(image_path))
    height,width=image.size(1),image.size(2)
    pixel_loc=get_pixels(height,width)

    checkpoint_path= "D:\\IDL\\Miniproject\\refined\\weight\\simple_network_butterfly.pt"
    checkpoint = torch.load(checkpoint_path)
    model=ImageMLP()
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion= nn.MSELoss()

    with torch.no_grad():
        pred=model(pixel_loc).reshape(512,512,3)
        gt=np.transpose(image,(1,2,0))
        plt.subplot(1,2,1)
        plt.imshow(pred.detach().numpy())
        plt.title('Predicted Image')
        plt.subplot(1,2,2)
        plt.imshow(gt.detach().numpy())
        plt.title('Ground Truth')
        plt.show()
