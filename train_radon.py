from PIL import Image
from torchvision import transforms
from utils import psnr, get_pixels, radon
from SSIM_PIL import compare_ssim
import numpy as np
import torch
import torch.nn as nn
from model import RadonMLP
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import tifffile

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    image_path= input("Enter image path: ")
    log_dir="D:\\IDL\\Miniproject\\refined\\results"
    model_path="D:\\IDL\\Miniproject\\refined\\weight\\simple_network_fox.pt"
    num_epochs=200

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    transform = transforms.ToTensor()

    image= transform(Image.open(image_path))

    height,width=image.size(1),image.size(2)
    pixel_loc=get_pixels(height,width)

    model= RadonMLP()
    optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
    num_epochs=5000
    criterion= nn.MSELoss(reduction='mean')
    

    shepp_radon= np.load('/content/shepp.npy')
    shepp_radon_gt= tifffile.imread('refined\data\radon\shepp.tiff')
    shepp_radon=torch.tensor(shepp_radon)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        est=model(pixel_loc)
        est=est.reshape(512,512,1)
        radon_transform= radon(est,1)
        l=criterion(radon_transform,shepp_radon)
        l.backward()
        optimizer.step()

        if epoch%10==0:
            print("training loss: ", (epoch+1, l.item()))
            writer.add_scalar('Loss/train', l.item(), epoch)

        if epoch%25==0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': l.item()}, model_path)

        



