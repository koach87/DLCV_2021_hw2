import argparse
import random
import torch
import torch.nn as nn
import os
import numpy as np
import imageio

manualSeed=878
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( 110, 28 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(28 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(28 * 8, 28 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(28 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( 28 * 4, 28 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(28 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( 28 * 2, 28, 2, 1, 1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(True),
            nn.ConvTranspose2d( 28, 3, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def output(model,outpath,device):
    #images.shape : (amount_of_images, 3, 28, 28), type:numpy
    
    each_imgs = torch.empty((0,3,28,28),device=device)
    for cls in range(10):
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        cls = torch.full((100,),cls,device=device)
        # print(cls)

        noise = torch.randn(100, 100, 1, 1, device=device)
        # normalize
        onehot_label = torch.eye(10,device=device)[cls].view(-1, 10, 1, 1)
        noise = torch.cat((noise,onehot_label),dim=1)
        
        images = model(noise)
        each_imgs = torch.cat((each_imgs,images),dim=0)

    each_imgs = each_imgs.cpu().detach().numpy()
    # output amt = ncls * each_image_amout images

    # transpose shape to output image(amt, 3, 28, 28) ->(amt, 28, 28, 3)
    imgs_for_output = np.transpose(each_imgs,(0,2,3,1))
    imgs_for_output = ((imgs_for_output+1)*127.5).astype(np.uint8)

    
    for i,ele in enumerate(imgs_for_output):
        #shape i = (28,28,3)
        imageio.imsave(os.path.join(outpath,"{}_{:0>3d}.png".format(i//100,i%100+1)),
                        ele.astype(np.uint8))
    

    
parser =  argparse.ArgumentParser(description='output GAN image')
parser.add_argument( '--output_path', type=str, default='', help='path to image' )
args = parser.parse_args()
output_path = args.output_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('p2.pth', map_location=device).to(device)
output(model,output_path,device)