#this script trains 2D image encoders from torchvision on a dataset of images loaded in from a custom dataloader
from slice_dataloader import BrainMRI_Dataset,get_data_loaders
import torch
import torchvision
from torch.utils.data import DataLoader

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
swin = torchvision.models.swin_v2_s(torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1)
cnn = torchvision.models.efficientnet_v2_s(torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
print(dev)

swin = swin.to(dev)
cnn = cnn.to(dev)

slices_fp = '/home/nebius/data/slices/slices'

tr_dl,vl_dl = get_data_loaders(slices_fp,'slice_meta.csv',batch_size=32,num_workers=10)

for im,lb in tr_dl:
    im,lb = im.to(dev),lb.to(dev)
    print(im.shape,lb.shape)
    inf = swin(im)
    print(inf.shape)
    print(inf)
    assert False