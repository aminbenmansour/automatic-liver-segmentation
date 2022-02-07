
from monai.networks.nets import UNet
from monai.networks.layers import Norm

import torch
from preporcess import prepare

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
