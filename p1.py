import argparse
import random
import torch
import torch.nn as nn
import os
import numpy as np
import imageio


path_test = '~/hw2/GAN/output_images'
manualSeed=999

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def output(model,out_amount,outpath,device):
    #images.shape : (amount_of_images, 3, 512, 512), type:numpy
    
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    noise = torch.randn(out_amount, 100, 1, 1, device=device)
    images = model(noise).cpu().detach().numpy()

    #transpose shape to output image(amt, 3, 512, 512) ->(amt, 512, 512, 3)
    imgs_for_output = np.transpose(images,(0,2,3,1))
    imgs_for_output = (imgs_for_output+1)*128
    for i,ele in enumerate(imgs_for_output):
        #shape i = (512,512,3)
        imageio.imsave(os.path.join(outpath,f'{i:04d}.png'),ele.astype(np.uint8))
    
    
parser =  argparse.ArgumentParser(description='output GAN image')
parser.add_argument( '--output_path', type=str, default='', help='path to image' )
args = parser.parse_args()
output_path = args.output_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('p1.pth', map_location=device).to(device)
output(model,1000,output_path,device)