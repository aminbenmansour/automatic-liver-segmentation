
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss

import torch
from preporcess import prepare
from utilities import calculate_weights

data_dir = './datasets/Data_Train_Test'
model_dir = './results/' 
data_in = prepare(data_dir, cache=True)


device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))