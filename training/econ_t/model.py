
import os
import sys 
sys.path.append(os.environ['HAWQ_JET_TAGGING'])
import torch.nn as nn
from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantConv2d


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


encoder_type = {
    'baseline': (3, 8, 128),
    'large': (5, 32, 288),
    'small': (3, 1, 16),
}

########################################################################
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


########################################################################
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
