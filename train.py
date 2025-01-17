#this script trains 2D image encoders from torchvision on a dataset of images loaded in from a custom dataloader
from slice_dataloader import BrainMRI_Dataset,get_data_loaders
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
swin = torchvision.models.swin_v2_s(torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1)
cnn = torchvision.models.efficientnet_v2_s(torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
swin.head = torch.nn.Linear(768,8)
cnn.classifier[1] = torch.nn.Linear(1280,8)

for param in swin.parameters():
    param.requires_grad = False
for param in swin.head.parameters():
    param.requires_grad = True

for param in cnn.parameters():
    param.requires_grad = False
for param in cnn.classifier[1].parameters():
    param.requires_grad = True

swin = swin.to(dev)
cnn = cnn.to(dev)



slices_fp = '/home/nebius/data/slices/slices'
epochs = 100
penultimate_layer_on = 101
tr_dl,vl_dl = get_data_loaders(slices_fp,'slice_meta.csv',batch_size=32,num_workers=10)

df = tr_dl.dataset.data_df
cnn_optimizer = torch.optim.Adam(cnn.parameters())
vit_optimizer = torch.optim.Adam(swin.parameters())
loss = torch.nn.CrossEntropyLoss()

cnn_losses = []
vit_losses = []

for epoch in range(epochs):
    if epoch == penultimate_layer_on:
        print('Unfreezing penultimate layer in each encoder')
        for param in cnn.features[7].parameters():
            param.requires_grad = True
        for param in swin.features[7][1].parameters():
            param.requires_grad = True

    cnn.train(),swin.train()
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

    cnn.eval(),swin.eval()
    # eval validation loss
    val_loss = [0,0]
    for i,(im,lb) in enumerate(vl_dl):
        im, lb = im.to(dev), lb.to(dev)
        lb = torch.nn.functional.one_hot(lb, num_classes=8).float()
        cnn_pred, vit_pred = cnn(im), swin(im)
        cnn_loss, vit_loss = loss(cnn_pred, lb), loss(vit_pred, lb)

        val_loss[0]+=cnn_loss.item()
        val_loss[1]+=vit_loss.item()
    cnn_optimizer.zero_grad(),vit_optimizer.zero_grad()
    val_loss = [loss/len(vl_dl) for loss in val_loss]
    cnn_losses.append(val_loss[0])
    vit_losses.append(val_loss[1])
    plt.close()
    fig = plt.figure()
    plt.plot(cnn_losses,label='CNN')
    plt.plot(vit_losses,label='Swin')
    plt.legend()
    plt.savefig('/home/nebius/hackathon/plot.png')
    plt.show()
    print('Val Losses on Epoch {}: CNN: {:.3f}, Swin: {:.3f}'.format(epoch, cnn_loss.item(), vit_loss.item()))
    print()
