"""
Compute Bit Operations (BOPs) for PyTorch and HAWQ models.
Supported layers:
    - Linear
    - Conv2d 
    - QuantLinear 
    - QuantConv2d 

todo:
    - add Batchnormalization, QuantBnConv2d
    - pytest
"""
import sys
sys.path.append('..')
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from hawq.utils.quantization_utils.quant_modules import QuantLinear, QuantConv2d, QuantAct



########################################################
# Helper Funcs
########################################################
# def countNonZeroWeights(model):
#     nonzero = total = 0
#     layer_count_alive = {}
#     layer_count_total = {}
#     for name, p in model.named_parameters():
#         tensor = p.data.cpu().numpy()
#         nz_count = np.count_nonzero(tensor)
#         total_params = np.prod(tensor.shape)
#         layer_count_alive.update({name: nz_count})
#         layer_count_total.update({name: total_params})
#         nonzero += nz_count
#         total += total_params
#         print(f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}")
#     print(f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)")
#     return nonzero, total, layer_count_alive, layer_count_total


def countNonZero(tensor, name):
    nz_count = np.count_nonzero(tensor)
    total_params = np.prod(tensor.shape)
    print(f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}")
    return nz_count, total_params


def countNonZeroWeights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, QuantLinear, QuantConv2d)):
            module_name = f"{name}.weight"
            tensor = module.weight.data.cpu().numpy()
            nz_count, total_params = countNonZero(tensor, module_name)
            layer_count_alive.update({module_name: nz_count})
            layer_count_total.update({module_name: total_params})
            nonzero += nz_count
            total += total_params
            if hasattr(module, "bias"):
                module_name = f"{name}.bias"
                tensor = module.bias.data.cpu().numpy()
                nz_count, total_params = countNonZero(tensor, module_name)
                layer_count_alive.update({module_name: nz_count})
                layer_count_total.update({module_name: total_params})
                nonzero += nz_count
                total += total_params
    print(f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)")
    return nonzero, total, layer_count_alive, layer_count_total, (100 * (total - nonzero) / total)


def forwardHook(module, input, output):
    if len(output) == 2:
        output, _ = output
    print(f"[{module.__class__.__name__}] Forward hook: Input shape: {input[0].shape}, Output shape: {output.shape}")
    module.input_shape = input[0].shape
    module.output_shape = output.shape


def registerHooks(model):
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            continue  
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.Flatten)):
            hooks.append(module.register_forward_hook(forwardHook))
        elif isinstance(module, (QuantLinear, QuantConv2d)):
            hooks.append(module.register_forward_hook(forwardHook))
    return hooks


def removeHooks(hooks):
    for hook in hooks:
        hook.remove()


def getInputShapes(model, input_shape):
    hooks = registerHooks(model)
    model(torch.randn(input_shape))
    removeHooks(hooks)


class QDenseModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.weight_precision = 8
        self.activation_precision = 8
        self.quant_act = QuantAct(self.activation_precision)
        self.init_dense(model, "0", self.weight_precision)
        self.init_dense(model, "2", self.weight_precision)
        self.init_dense(model, "4", self.weight_precision)
        self.init_dense(model, "6", self.weight_precision)
    
    def init_dense(self, module, name, bw):
        layer = getattr(module, name)
        quant_layer = QuantLinear(weight_bit=bw, bias_bit=bw)
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)

    def forward(self, x):
        x, p_sf = self.quant_act(x)
        x = getattr(self, '0')(x, p_sf)
        x = getattr(self, '2')(x, p_sf)
        x = getattr(self, '4')(x, p_sf)
        x = getattr(self, '6')(x, p_sf)


class QConvModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.weight_precision = 7
        self.activation_precision = 7
        self.quant_act = QuantAct(self.activation_precision)
        self.init_conv2d(model, "0", self.weight_precision)
        self.init_conv2d(model, "2", self.weight_precision)
        self.init_dense(model, "6", self.weight_precision)
        self.init_dense(model, "8", self.weight_precision)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten(start_dim=0)

    def init_conv2d(self, module, name, bw):
        layer = getattr(module, name)
        quant_layer = QuantConv2d(weight_bit=bw, bias_bit=bw)
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)

    def init_dense(self, module, name, bw):
        layer = getattr(module, name)
        quant_layer = QuantLinear(weight_bit=bw, bias_bit=bw)
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)
    
    def forward(self, x):
        x, p_sf = self.quant_act(x)
        x, w_sf = getattr(self, '0')(x, p_sf)
        x, w_sf = getattr(self, '2')(x, p_sf)
        x = self.pool(x)
        x = self.flatten(x)
        x = getattr(self, '6')(x, p_sf)
        x = getattr(self, '8')(x, p_sf)
        return x


########################################################
# Compute BOPs
########################################################
def compute_bops(model, input_shape):
    getInputShapes(model, input_shape)

    # assume 32 bit precision 
    b_w = 32
    b_a = 32
    alive, total, l_alive, l_total, pruned = countNonZeroWeights(model)
    total_bops = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n = module.in_features
            m = module.out_features
            k = 1
        elif isinstance(module, nn.Conv2d):
            n = np.prod(module.input_shape) 
            m = np.prod(module.output_shape) 
            k = np.prod(module.kernel_size)
        else:
            continue
        
        total = l_total[name + ".weight"] + l_total[name + ".bias"]
        alive = l_alive[name + ".weight"] + l_alive[name + ".bias"]
        p = 1 - ((total - alive) / total)  # fraction of layer remaining
        
        # assuming b_a is the output bitwidth of the last layer
        # module_bops = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
        module_bops = m * n * k * k *(p * b_a * b_w + b_a + b_w + math.log2(n*k*k))
        print("{} BOPS: {} = {}*{}*{}({}*{}*{} + {} + {} + {})".format(name,module_bops,m,n,k*k,p,b_a,b_w,b_a,b_w,math.log2(n*k*k)))

        total_bops += module_bops
    print("Total BOPS: {}".format(total_bops))
    return total_bops


def computeBOPsHAWQ(model, input_data_precision=32, input_shape=None):
    getInputShapes(model, input_shape)

    last_bit_width = input_data_precision
    alive, total, l_alive, l_total, pruned = countNonZeroWeights(model)
    # assume same weight precision everywhere 
    b_w = model.weight_precision if hasattr(model, "weight_precision") else 32  
    total_bops = 0

    for name, module in model.named_modules():
        b_a = last_bit_width
        if isinstance(module, QuantLinear):
            n = module.in_features
            m = module.out_features
            k = 1
        elif isinstance(module, QuantConv2d):
            n = np.prod(module.input_shape) 
            m = np.prod(module.output_shape) 
            k = np.prod(module.kernel_size)
        else:
            continue 

        total = l_total[name + ".weight"] + l_total[name + ".bias"]
        alive = l_alive[name + ".weight"] + l_alive[name + ".bias"]
        p = 1 - ((total - alive) / total)  # fraction of layer remaining
        p = 1

        # assuming b_a is the output bitwidth of the last layer
        # module_bops = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
        module_bops = m * n * k * k *(p * b_a * b_w + b_a + b_w + math.log2(n*k*k))
        print("{} BOPS: {} = {}*{}*{}({}*{}*{} + {} + {} + {})".format(name,module_bops,m,n,k*k,p,b_a,b_w,b_a,b_w,math.log2(n*k*k)))
        last_bit_width = b_w
        total_bops += module_bops
    print("Total BOPS: {}".format(total_bops))
    return total_bops



if __name__ == "__main__":

    ####################
    # MLP 
    ####################
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
        nn.Softmax()
    )

    # FP32 Model 
    total_bops = compute_bops(model, input_shape=[1, 16])
    assert total_bops == 4652832, f'Error: Incorrect BOPs count {total_bops} for FP32 model'

    # INT8 Model 
    qmodel = QDenseModel(model)
    total_bops = computeBOPsHAWQ(qmodel, input_data_precision=8, input_shape=[1, 16])

    ####################
    # CNN
    ####################
    height, width = 8, 8

    # Define the CNN model using Sequential
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(start_dim=0),
        nn.Linear(64 * (height // 4) * (width // 4), 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # FP32 Model 
    compute_bops(model, input_shape=[1, 8, 8])

    # INT7 Model 
    qmodel = QConvModel(model)
    total_bops = computeBOPsHAWQ(qmodel, input_data_precision=32, input_shape=[1, 8, 8])
    assert total_bops == 9244014213.550673, f'Error: Incorrect BOPs count {total_bops} for model'
