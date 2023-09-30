import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from utils import psnr, get_pixels, get_fourier_features
from SSIM_PIL import compare_ssim
from model import Fourier_MLP
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    image_path= input("Enter image path: ")
    log_dir="D:\\IDL\\Miniproject\\refined\\results"
    model_path="D:\\IDL\\Miniproject\\refined\\weight\\fourier_fox.pt"
    num_epochs=200

    transform = transforms.ToTensor()

    image= transform(Image.open(image_path))

    height,width=image.size(1),image.size(2)
    pixel_loc=get_pixels(height,width)
    fourier_transform= get_fourier_features(pixel_loc,32)
    fourier_transform= torch.tensor(fourier_transform,dtype=torch.float)

    model=Fourier_MLP()
    optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
    num_epochs=300
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


