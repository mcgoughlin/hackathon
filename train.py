#this script trains 2D image encoders from torchvision on a dataset of images loaded in from a custom dataloader
from slice_dataloader import BrainMRI_Dataset,get_data_loaders
import torch
import torchvision
from torch.utils.data import DataLoader

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
swin = torchvision.models.swin_v2_s(torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1)
cnn = torchvision.models.efficientnet_v2_s(torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
swin.head = torch.nn.Linear(768,8)
cnn.classifier[1] = torch.nn.Linear(1280,8)

swin = swin.to(dev)
cnn = cnn.to(dev)

slices_fp = '/home/nebius/data/slices/slices'
epochs = 100
tr_dl,vl_dl = get_data_loaders(slices_fp,'slice_meta.csv',batch_size=32,num_workers=10)

df = tr_dl.dataset.data_df
cnn_optimizer = torch.optim.Adam(cnn.parameters())
vit_optimizer = torch.optim.Adam(swin.parameters())
loss = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for i,(im,lb) in enumerate(tr_dl):
        im,lb = im.to(dev),lb.to(dev)
        lb = torch.nn.functional.one_hot(lb,num_classes=8).float()

        cnn_pred,vit_pred = cnn(im),swin(im)
        cnn_loss,vit_loss = loss(cnn_pred,lb), loss(vit_pred,lb)

        cnn_loss.backward(),vit_loss.backward()
        cnn_optimizer.step(),vit_optimizer.step()
        cnn_optimizer.zero_grad(),vit_optimizer.zero_grad()
        if i==0:
            print('Epoch {}: CNN Loss: {:.3f}, Swin Loss: {:.3f}'.format(epoch,cnn_loss.item(),vit_loss.item()))