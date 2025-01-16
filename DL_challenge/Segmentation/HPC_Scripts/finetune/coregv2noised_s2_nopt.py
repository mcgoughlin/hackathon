import os

os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data/"
from KCD.Segmentation.ovseg.model.SegmentationModel import SegmentationModel
from KCD.Segmentation.ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import gc
import torch
import sys
from KCD.Segmentation.Inference import infer_network

def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]

    return infer_network.UNet_ClassHead_PowerPool_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

data_name = 'masked_coreg_v2_noised'
spacing = 2
fold = int(sys.argv[1])

# preprocessed_name = '4mm_binary'
preprocessed_name = '2mm_cancerbinary_customPP'
# model_name = 'del'

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
vfs = [fold]

patch_size = [64, 64, 64]
# patch dimension must be divisible by respective (((kernel_dimension+1)//2)^depth)/2
# Patch size dictates input size to CNN: input dim (metres) = patch_size*target_spacing/1000
# finally, depth and conv kernel size dictate attentive area - importantly different to input size:
# attentive_area (in each dimension, metres) = input size / bottom encoder spatial dim
#                                           = ((((kernel_dimension+1)//2)^depth)/2)*target_spacing/1000
z_to_xy_ratio = 1
larger_res_encoder = True
n_fg_classes = 1

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=False, fp32=True)

model_params['architecture'] = 'UNet'
model_params['training']['decoder_frozen'] = 0
model_params['training']['encoder_frozen'] = 0

model_params['network']['kernel_sizes'] = 6 * [(3, 3, 3)]
model_params['network']['norm'] = 'inst'
model_params['network']['in_channels'] = 1
model_params['network']['filters'] = 32
model_params['network']['filters_max'] = 1024
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']
del model_params['training']['loss_params']

lr = 0.000001
model_params['data']['folders'] = ['images', 'labels']
model_params['data']['keys'] = ['image', 'label']
model_params['training']['num_epochs'] = 150
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': lr,
                                          'betas': (0.95, 0.9),
                                          'eps': 1e-08}
model_params['training']['lr_params'] = {'n_warmup_epochs': 25, 'lr_max': 0.00005}
model_params['data']['trn_dl_params']['epoch_len'] = 1000
model_params['data']['trn_dl_params']['padded_patch_size'] = [2 * patch_size[0]] * 3
model_params['data']['val_dl_params']['padded_patch_size'] = [2 * patch_size[0]] * 3
model_params['training']['lr_schedule'] = 'lin_ascent_log_decay'
model_params['training']['lr_exponent'] = 2
model_params['data']['trn_dl_params']['batch_size'] = 32
model_params['data']['val_dl_params']['epoch_len'] = 200
model_params['data']['trn_dl_params']['min_biased_samples'] = model_params['data']['trn_dl_params']['batch_size'] //2

# preprocessed_name='4mm_binary_test'
model_name = '6_nopretrain'

model_name = model_name + '_seg'
for vf in vfs:

    model = SegmentationModel(val_fold=vf,
                              data_name=data_name,
                              preprocessed_name=preprocessed_name,
                              model_name=model_name,
                              model_parameters=model_params)
    # model.network.load_state_dict(sd)

    model.training.train()
    model.eval_validation_set(continuous=False)
