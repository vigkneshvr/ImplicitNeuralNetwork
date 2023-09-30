import numpy as np
import torch
import torchvision

psnr = lambda gt , est : -20*np.log10(np.linalg.norm((gt-est).reshape(-1))/(np.sqrt(gt.size)*np.max(gt)))

def get_pixels(height, width):
    x,y=np.meshgrid(np.linspace(0,1,height,dtype=np.float32),np.linspace(0,1,width,dtype=np.float32))
    x=x.reshape(height*width,1)
    y=y.reshape(height*width,1)
    return torch.tensor(np.hstack((x,y)),dtype=torch.float)

def get_fourier_features(c, k):
    b = np.random.normal(size=(k, 2))
    # freqs /= np.linalg.norm(freqs, axis=1, keepdims=True)
    x=np.cos(2 * np.pi * np.dot(c, b.T))
    y=np.sin(2 * np.pi * np.dot(c, b.T))
    return np.concatenate((x, y), axis=1)

def radon(image,channels):
  radon_list = []
  image=image.permute(2,0,1)
  for angle in range(180):
      rotated_image = torchvision.transforms.functional.rotate(image, angle)
      s = torch.sum(rotated_image, 1)
      radon_list.append(s)
  radon = torch.stack(radon_list, 1)
  radon=radon.squeeze(2)
  radon=radon.permute(2,1,0)
  return radon

def nlradon_transform(img,channels):
    radon_lis = []
    for degree in range(360):
        rotated = torchvision.transforms.functional.rotate(img.permute(2, 0, 1), degree)
        coeff = torch.exp(-0.01 * torch.cumsum(rotated, 1))
        integral = torch.sum(rotated * coeff, 1, keepdim=True)
        radon_lis.append(integral)
    radon = torch.stack(radon_lis, 1).permute(2, 3, 1, 0)[0]
    return radon