import sys
sys.path.append('..')
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from hawq.utils.quantization_utils.quant_modules import QuantLinear



########################################################
# Helper Funcs
########################################################
def countNonZeroWeights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        layer_count_alive.update({name: nz_count})
        layer_count_total.update({name: total_params})
        nonzero += nz_count
        total += total_params
        print(f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}")
    print(f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)")
    return nonzero, total, layer_count_alive, layer_count_total


def countNonZeroIntergerWeights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    for name, p in model.named_buffers():
        if "integer" not in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        layer_count_alive.update({name: nz_count})
        layer_count_total.update({name: total_params})
        nonzero += nz_count
        total += total_params
        print(f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}")
    print(f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)")
    return nonzero, total, layer_count_alive, layer_count_total, (100 * (total - nonzero) / total)


class QModel(nn.Module):
        def __init__(self, model) -> None:
            super().__init__()

            self.weight_precision = 8
            self.activation_precision = 8

            self.init_dense(model, "0", self.weight_precision)
            self.init_dense(model, "2", self.weight_precision)
            self.init_dense(model, "4", self.weight_precision)
            self.init_dense(model, "6", self.weight_precision)
        
        def init_dense(self, module, name, bw):
            layer = getattr(module, name)
            quant_layer = QuantLinear(weight_bit=bw, bias_bit=bw)
            quant_layer.set_param(layer)
            setattr(self, name, quant_layer)


########################################################
# Compute BOPs
########################################################
def compute_bops(model):
    # assume 32 bit precision 
    b_w = 32
    b_a = 32
    alive, total, l_alive, l_total = countNonZeroWeights(model)
    total_bops = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n = module.in_features
            m = module.out_features
        else:
            continue
        total = l_total[name + ".weight"] + l_total[name + ".bias"]
        alive = l_alive[name + ".weight"] + l_alive[name + ".bias"]
        p = 1 - ((total - alive) / total)  # fraction of layer remaining
        
        # assuming b_a is the output bitwidth of the last layer
        # module_bops = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
        module_bops = m * n * (p * b_a * b_w + b_a + b_w + math.log2(n))
        print("{} BOPS: {} = {}*{}*({}*{}*{} + {} + {} + {})".format(name, module_bops, m, n, p, b_a, b_w, b_a, b_w, math.log2(n)))

        total_bops += module_bops
    print("Total BOPS: {}".format(total_bops))
    return total_bops


def computeBOPsHAWQ(model, input_data_precision=32):
    last_bit_width = input_data_precision
    alive, total, l_alive, l_total, pruned = countNonZeroIntergerWeights(model)
    # assume same weight precision everywhere 
    b_w = model.weight_precision if hasattr(model, "weight_precision") else 32  
    total_bops = 0

    for name, module in model.named_modules():
        b_a = last_bit_width
        if isinstance(module, QuantLinear):
            n = module.in_features
            m = module.out_features
        else:
            continue 

        total = l_total[name + ".weight_integer"] + l_total[name + ".bias_integer"]
        alive = l_alive[name + ".weight_integer"] + l_alive[name + ".bias_integer"]
        p = 1 - ((total - alive) / total)  # fraction of layer remaining

        # assuming b_a is the output bitwidth of the last layer
        # module_bops = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
        module_bops = m * n * (p * b_a * b_w + b_a + b_w + math.log2(n))
        print("{} BOPS: {} = {}*{}({}*{}*{} + {} + {} + {})".format(name, module_bops, m, n, p, b_a, b_w, b_a, b_w, math.log2(n)))
        last_bit_width = b_w  
        total_bops += module_bops
    print("Total BOPS: {}".format(total_bops))
    return total_bops, pruned



if __name__ == "__main__":
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

    total_bops = compute_bops(model)
    assert total_bops == 4652832, 'Error: Incorrect BOPs count for FP32 model'


    qmodel = QModel(model)
    total_bops = computeBOPsHAWQ(qmodel, 8)
