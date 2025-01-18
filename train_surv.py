#this script trains 2D image encoders from torchvision on a dataset of images loaded in from a custom dataloader
from seg_dataloader import Seg_Dataset,get_data_loaders
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
swin = torchvision.models.swin_v2_s(torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1)
cnn = torchvision.models.efficientnet_v2_s(torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
swin.head = torch.nn.Linear(768,2)
cnn.classifier[1] = torch.nn.Linear(1280,2)

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

slices_fp = '/home/nebius/data/survival_slices/survival_slices'
epochs = 1
tr_dl,vl_dl = get_data_loaders(slices_fp,'slice_meta.csv',batch_size=128,num_workers=10)

df = tr_dl.dataset.data_df
cnn_optimizer = torch.optim.Adam(cnn.parameters(),lr=0.0001)
vit_optimizer = torch.optim.Adam(swin.parameters(),lr = 0.0001)
loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([2008/(2008-594),2008/594]).to(dev))

cnn_losses = []
vit_losses = []
block_index = 7
training_losses =[]
for epoch in range(epochs):
    cnn.train(),swin.train()
    for i,(_,im,seg,lb) in enumerate(tr_dl):
        im,lb = im.to(dev),lb.to(dev)
        lb = lb.long()
        cnn_pred,vit_pred = cnn(im),swin(im)
        lb.squeeze(), cnn_pred.squeeze(),vit_pred.squeeze()
        cnn_loss,vit_loss = loss(cnn_pred,lb), loss(vit_pred,lb)
        training_losses.append((cnn_loss.item()))

        cnn_loss.backward(),vit_loss.backward()
        cnn_optimizer.step(),vit_optimizer.step()
        cnn_optimizer.zero_grad(),vit_optimizer.zero_grad()

    # fig = plt.figure()
    # plt.plot(training_losses)
    # plt.savefig('/home/nebius/hackathon/plot_train.png')
    # plt.close()

    cnn.eval(),swin.eval()
    # eval validation loss
    val_loss = [0,0]
    correct_cnn = 0
    correct_vit = 0
    unique_means = {}
    for i,(fp,im,seg,lb) in enumerate(vl_dl):
        im, lb = im.to(dev), lb.to(dev).long()
        print(fp)
        fp_list = np.array([file[:6] for file in fp])
        unique_ids = np.unique(fp_list)
        print(len(np.unique(fp)))
        cnn_pred, vit_pred = cnn(im), swin(im)
        print(torch.nn.functional.softmax(cnn_pred,dim=-1))
        print(lb)
        cnn_loss, vit_loss = loss(cnn_pred, lb), loss(vit_pred, lb)
        # senses, specs = [], []
        # for thresh in np.arange(0,100,1):
        #     thresh/=100


        vit_pred_bin = torch.argmax(vit_pred,dim=-1)

        correct_vit+=torch.sum(vit_pred_bin==lb)

        vit_tp = ((vit_pred_bin==lb) & (lb==1)).sum()
        vit_tn = ((vit_pred_bin==lb) & ~(lb==1)).sum()
        vit_fp = ((vit_pred_bin!=lb) & (lb==1)).sum()
        vit_fn = ((vit_pred_bin!=lb) & ~(lb==1)).sum()

        val_loss[0]+=cnn_loss.item()
        val_loss[1]+=vit_loss.item()
    cnn_optimizer.zero_grad(),vit_optimizer.zero_grad()
    val_loss = [loss/len(vl_dl) for loss in val_loss]
    cnn_losses.append(val_loss[0])
    vit_losses.append(val_loss[1])
    fig = plt.figure()
    plt.plot(cnn_losses,label='CNN')
    plt.plot(vit_losses,label='Swin')
    plt.legend()

    vit_sens = vit_tp/(vit_tp+vit_fn)
    vit_spec = vit_tn/(vit_tn+vit_fp)

    plt.savefig('/home/nebius/hackathon/plot_surv.png')
    print('Val Losses on Epoch {}: CNN: {:.3f}, Swin: {:.3f}'.format(epoch, cnn_loss.item(), vit_loss.item()))
    print('  CNN: {:.1f}, Swin: {:.1f}'.format(100*correct_cnn/len(lb),100*correct_vit/len(lb)))
    print(vit_sens.item(),vit_spec.item())
    print()
