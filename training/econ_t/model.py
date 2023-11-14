import os
import sys 
sys.path.append('../..')
import torch
import torch.nn as nn
import pytorch_lightning as pl

from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantConv2d
from collections import OrderedDict
from telescope_pt import telescopeMSE8x8, move_constants_to_gpu
from autoencoder_datamodule import ARRANGE, ARRANGE_MASK


"""
Model: "encoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 8, 8, 1)]         0         
                                                                 
 input_qa (QActivation)      (None, 8, 8, 1)           0         
                                                                 
 conv2d_0_m (FQConv2D)       (None, 4, 4, 8)           80        
                                                                 
 accum1_qa (QActivation)     (None, 4, 4, 8)           0         
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 encoded_vector (FQDense)    (None, 16)                2064      
                                                                 
 encod_qa (QActivation)      (None, 16)                0         
                                                                 
=================================================================
Total params: 2,144
Trainable params: 2,144
Non-trainable params: 0

_________________________________________________________________
Model: "decoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 decoder_input (InputLayer)  [(None, 16)]              0         
                                                                 
 de_dense_final (Dense)      (None, 128)               2176      
                                                                 
 de_reshape (Reshape)        (None, 4, 4, 8)           0         
                                                                 
 conv2D_t_0 (Conv2DTranspose  (None, 8, 8, 8)          584       
 )                                                               
                                                                 
 conv2d_t_final (Conv2DTrans  (None, 8, 8, 1)          73        
 pose)                                                           
                                                                 
 decoder_output (Activation)  (None, 8, 8, 1)          0         
                                                                 
=================================================================
Total params: 2,833
Trainable params: 2,833
Non-trainable params: 0
"""


CALQ_MASK = torch.tensor(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    ]
)

encoder_type = {
    'baseline': (3, 8, 128),
    'large': (5, 32, 288),
    'small': (3, 1, 16),
}


class BaseEncoder(nn.Module):
    def __init__(self, econ_type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoded_dim = 16
        self.shape = (1, 8, 8)  # PyTorch defaults to (C, H, W)
        self.val_sum = None

        kernel_size, num_kernels, fc_input = encoder_type[econ_type]

        self.conv = nn.Conv2d(1, num_kernels, kernel_size=kernel_size, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.enc_dense = nn.Linear(fc_input, self.encoded_dim)
    
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.enc_dense(x))
        return x

class QuantizedEncoder(nn.Module):
    def __init__(self, model, precision) -> None:
        super().__init__()

        self.quant_input = QuantAct(activation_bit=16)
        self.quant_relu = QuantAct(activation_bit=precision[0]+3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()

        base_layer = getattr(model, 'conv')
        hawq_layer = QuantConv2d(precision[0], precision[0])
        hawq_layer.set_param(base_layer)
        setattr(self, 'conv', hawq_layer)

        base_layer = getattr(model, 'enc_dense')
        hawq_layer = QuantLinear(precision[1], precision[1])
        hawq_layer.set_param(base_layer)
        setattr(self, 'enc_dense', hawq_layer)
    
    def forward(self, x):
        x, p_sf = self.quant_input(x)

        x, w_sf = self.conv(x, p_sf)
        x = self.relu1(x)
        x, p_sf = self.quant_relu(x, p_sf, w_sf)

        x = self.flatten(x)
        x = self.relu2(self.enc_dense(x, p_sf))
        return x


class AutoEncoder(pl.LightningModule):
    def __init__(self, accelerator, quantize, precision, learning_rate, econ_type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoded_dim = 16
        self.shape = (1, 8, 8)  # PyTorch defaults to (C, H, W)
        self.val_sum = None
        self.quantize = quantize
        self.accelerator = accelerator
        self.learning_rate = learning_rate

        self.encoder = BaseEncoder(econ_type) 
        if self.quantize:
            self.encoder = QuantizedEncoder(self.encoder, precision=precision)
        
        self.decoder = nn.Sequential(OrderedDict([
            ("dec_dense", nn.Linear(self.encoded_dim, 128)),
            ("relu1", nn.ReLU()),
            ("unflatten", nn.Unflatten(1, (8, 4, 4))),
            (
                "convtrans2d1",
                nn.ConvTranspose2d(
                    8, 8, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
            ),
            ("relu2", nn.ReLU()),
            (
                "convtrans2d2",
                nn.ConvTranspose2d(
                    8,
                    self.shape[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
            ("sigmoid", nn.Sigmoid()),
        ]))
        self.loss = telescopeMSE8x8

    def invert_arrange(self):
        """
        Invert the arrange mask
        """
        remap = []
        hashmap = {}  # cell : index mapping
        found_duplicate_charge = len(ARRANGE[ARRANGE_MASK == 1]) > len(
            torch.unique(ARRANGE[ARRANGE_MASK == 1])
        )
        for i in range(len(ARRANGE)):
            if ARRANGE_MASK[i] == 1:
                if found_duplicate_charge:
                    if CALQ_MASK[i] == 1:
                        hashmap[int(ARRANGE[i])] = i
                else:
                    hashmap[int(ARRANGE[i])] = i
        for i in range(len(torch.unique(ARRANGE))):
            remap.append(hashmap[i])
        return torch.tensor(remap)

    def map_to_calq(self, x):
        """
        Map the input/output of the autoencoder into CALQs orders
        """
        remap = self.invert_arrange()
        image_size = self.shape[0] * self.shape[1] * self.shape[2]
        reshaped_x = x.reshape(len(x), image_size)
        reshaped_x[:, ARRANGE_MASK == 0] = 0
        return reshaped_x[:, remap]

    def set_val_sum(self, val_sum):
        self.val_sum = val_sum

    def predict(self, x):
        decoded_Q = self(x)
        encoded_Q = self.encoder(x)
        encoded_Q = torch.reshape(encoded_Q, (len(encoded_Q), self.encoded_dim, 1))
        return decoded_Q, encoded_Q

    # Pytorch Lightning specific methods
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # lr=1e-3
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss(x, x_hat)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss(x, x_hat)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
