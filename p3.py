import argparse
from PIL import Image
import torch
import torch.nn as nn
import glob
import os
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import torchvision.transforms as transforms

# reference from https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class DATA(Dataset):
    def __init__(self, img_path,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])) :
        self.img_path = img_path
        self.transform = transform
        self.filepaths = []
        self.filenames = []
        fns = glob.glob(os.path.join(img_path+'/*.png'))
        for i in fns:
            self.filepaths.append(i)
            self.filenames.append(os.path.basename(i))
        self.len = len(self.filepaths)
    
    def __getitem__(self, index) :
        fn = self.filepaths[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.filenames[index]

    def __len__(self):
        return self.len

    def get_filenames(self):
        return self.filenames

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 28, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(28, 28 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(28 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(28 * 2, 28 * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(28 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(28 * 4, 28 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(28 * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.realfake = nn.Sequential(
            nn.Conv2d(28 * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.Conv2d(28 * 8, 10, 2, 1, 0, bias=False),
            nn.Softmax()
        )

    def forward(self, input):
        input = self.main(input)
        rev_input = grad_reverse(input)
        dt = self.realfake(rev_input)
        cls = self.cls(input)
        return dt ,cls


def output(model,test_dataloader, csv_path):
    print('start output')
    labels = []
    fns = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data,fn in test_dataloader:
        output = model(data.to(device))[1].view(-1,10)
        pred = output.max(1, keepdim=False)[1].tolist()
        labels += pred
        fns += [i for i in fn]
    df = pd.DataFrame(
        {
            "image_name":fns,
            "label":labels
        }
    )
    # print(fns)
    # print(labels)
    df.to_csv(csv_path,index = False)

parser =  argparse.ArgumentParser(description='Use to predict image for HW1_p1')
parser.add_argument( '--img_path', type=str, default='', help='path to testing images in the target domain' )
parser.add_argument( '--traget_name', type=str, default='', help='a string that indicates the name of the target domain,which will be either mnistm, usps or svhn.')
parser.add_argument( '--csv_path', type=str, default='', help='path to your output prediction file')
args = parser.parse_args()

img_path = args.img_path
traget_name = args.traget_name
csv_path = args.csv_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = DATA(img_path)
test_dataloader = DataLoader(test_dataset, batch_size=128)
model = torch.load('{}.pth'.format(traget_name), map_location=device).to(device)
print('start predicting')
output(model, test_dataloader,csv_path)
print('done')