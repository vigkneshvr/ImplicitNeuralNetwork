from PIL import Image
from torchvision import transforms
from utils import psnr, get_pixels
from SSIM_PIL import compare_ssim
import numpy as np
import torch
import torch.nn as nn
from model import ImageMLP
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    image_path= input("Enter image path: ")
    log_dir="D:\\IDL\\Miniproject\\refined\\results"
    model_path="D:\\IDL\\Miniproject\\refined\\weight\\simple_network_fox.pt"
    num_epochs=200

    transform = transforms.ToTensor()

    image= transform(Image.open(image_path))

    height,width=image.size(1),image.size(2)
    pixel_loc=get_pixels(height,width)

    model=ImageMLP()
    optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
    criterion= nn.MSELoss()

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        target=np.transpose(image,(1,2,0))
        optimizer.zero_grad()
        est=model(pixel_loc)
        est=est.reshape(512,512,3)
        l=criterion(est,target)
        ps=psnr(target.detach().numpy(),est.detach().numpy())
        ssim = compare_ssim(Image.fromarray((est.detach().numpy()).astype(np.uint8)),Image.fromarray((target.detach().numpy()).astype(np.uint8)))
        l.backward()
        optimizer.step()
        if epoch%10==0:
            print("training loss: ", (epoch+1, l.item()))
            writer.add_scalar('Loss/train', l.item(), epoch)
            writer.add_scalar('psnr', ps, epoch)
            writer.add_scalar('ssim', ssim, epoch)
        
        if epoch%25==0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': l.item()}, model_path)

