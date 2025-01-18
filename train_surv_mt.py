#this script trains 2D image encoders from torchvision on a dataset of images loaded in from a custom dataloader
from seg_dataloader import Seg_Dataset,get_data_loaders
import torch
from torchvision import models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from modified_efficient_net import get_modified_efficientnet
import os
import numpy as np
torch.manual_seed(0)
np.random.seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

cnn = get_modified_efficientnet()
cnn.classifier = torch.nn.Linear(1280,2)

cnn = cnn.to(dev)

slices_fp = '/home/nebius/data/survival_slices/survival_slices'
epochs = 10
epoch_save = 1000
seg_weight = 0
tr_dl,vl_dl = get_data_loaders(slices_fp,'slice_meta.csv',batch_size=128,num_workers=10)

df = tr_dl.dataset.data_df
cnn_optimizer = torch.optim.Adam(cnn.parameters(),lr=0.0001)
loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([2008/(2008-594),2008/594]).to(dev))
seg_loss = torch.nn.CrossEntropyLoss()

cnn_losses = []
vit_losses = []
block_index = 7
for epoch in range(epochs):
    # if block_index>0:
    #     if (epoch % layer_turn_on == 0) and epoch != 0:
    #         #loop backwards through model.features and turn on gradients
    #         for param in cnn.features[block_index].parameters():
    #             param.requires_grad = True
    #
    #         print(epoch)
    #         print('Opened up the next layer for training:',block_index)
    #         block_index -=1

    cnn.train()
    for i,(_,im,seg,lb) in enumerate(tr_dl):
        im,lb,seg = im.to(dev),lb.to(dev),seg.to(dev)
        seg = seg.long()
        lb = lb.long()
        cnn_pred,seg_pred = cnn(im)
        lb.squeeze(), cnn_pred.squeeze(), seg.squeeze()
        cnn_loss = loss(cnn_pred,lb) + seg_weight*seg_loss(seg_pred,seg)

        cnn_loss.backward()
        cnn_optimizer.step()
        cnn_optimizer.zero_grad()
        # if i==0:
        # print('Epoch {}: CNN Loss: {:.3f}, Swin Loss: {:.3f}'.format(epoch,cnn_loss.item(),vit_loss.item()))

    cnn.eval()
    # eval validation loss
    val_loss = [0, 0]
    correct_cnn=0
    correct_vit = 0
    for i,(_,im,seg,lb) in enumerate(vl_dl):
        im, lb, seg = im.to(dev), lb.to(dev), seg.to(dev).long()
        cnn_pred,seg_pred= cnn(im)
        lb = lb.long()
        surv_loss_val = loss(cnn_pred, lb)
        seg_loss_val = seg_loss(seg_pred,seg)
        cnn_pred_bin = torch.argmax(cnn_pred,dim=-1)

        cnn_tp = ((cnn_pred_bin==lb) & (lb==1)).sum()
        cnn_tn = ((cnn_pred_bin==lb) & ~(lb==1)).sum()
        cnn_fp = ((cnn_pred_bin!=lb) & (lb==1)).sum()
        cnn_fn = ((cnn_pred_bin!=lb) & ~(lb==1)).sum()

        val_loss[0]+= surv_loss_val.item()
        val_loss[1]+= seg_loss_val.item()
        correct_cnn += torch.sum(cnn_pred_bin==lb)

        if epoch%epoch_save == 0:
            home_save = '/home/nebius/data/plots/epoch{}'.format(epoch)
            if not os.path.exists(home_save):
                os.makedirs(home_save)
            for sidx in range(len(im)):
                slice_im = im[sidx]
                slice_seg=seg[sidx]
                fig = plt.figure()
                plt.subplot(121)
                plt.imshow(slice_im.cpu().numpy().T)
                plt.subplot(122)
                plt.imshow(slice_seg.cpu().numpy().T)
                plt.savefig(os.path.join(home_save,'{}.png'.format(sidx)))
                plt.close()

    cnn_sens = cnn_tp / (cnn_tp + cnn_fn)
    cnn_spec = cnn_tn / (cnn_tn + cnn_fp)
    cnn_optimizer.zero_grad()
    val_loss = [loss/len(vl_dl) for loss in val_loss]
    cnn_losses.append(val_loss[0])
    vit_losses.append(val_loss[1])
    fig = plt.figure()
    plt.plot(cnn_losses,label='MSE')
    plt.plot(vit_losses,label='CCE')
    plt.legend()
    plt.savefig('/home/nebius/hackathon/plot_surv_mt.png')
    print('Val Losses on Epoch {}: SURV: {:.3f}, SEG {:.3f}'.format(epoch, val_loss[0], val_loss[1]))
    print('  Accuracy: {:.1f}, Sensitivity: {:.1f}, Specificity: {:.1f}'.format(100*correct_cnn/len(lb),
                                                                                cnn_sens*100,
                                                                                cnn_spec*100))
    print()
