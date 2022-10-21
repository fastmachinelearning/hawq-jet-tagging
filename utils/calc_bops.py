import math
import logging

import torch
import numpy as np
from hawq.utils.quantization_utils.quant_modules import QuantLinear


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
        print(
            f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
        )
    print(
        f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
    )
    return nonzero, total, layer_count_alive, layer_count_total


def calc_BOPS(model, input_data_precision=32):
    last_bit_width = input_data_precision
    alive, total, l_alive, l_total = countNonZeroWeights(model)
    b_w = model.weight_precision if hasattr(model, "weight_precision") else 32
    total_BOPS = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            b_a = last_bit_width
            # b_w = module.quant_weight_bit_width #Dont think this is a property I can access sadly, going with precision as given set in model
            n = module.in_features
            m = module.out_features
            total = l_total[name + ".weight"] + l_total[name + ".bias"]
            alive = l_alive[name + ".weight"] + l_alive[name + ".bias"]
            p = 1 - ((total - alive) / total)  # fraction of layer remaining
            # assuming b_a is the output bitwidth of the last layer
            # module_BOPS = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
            module_BOPS = m * n * (p * b_a * b_w + b_a + b_w + math.log2(n))
            print(
                "{} BOPS: {} = {}*{}({}*{}*{} + {} + {} + {})".format(
                    name, module_BOPS, m, n, p, b_a, b_w, b_a, b_w, math.log2(n)
                )
            )
            last_bit_width = b_w
            total_BOPS += module_BOPS
    print("Total BOPS: {}".format(total_BOPS))


def check_layers(name):
    layers = [
        "dense_1.weight_integer",
        "dense_1.bias_integer",
        "dense_2.weight_integer",
        "dense_2.bias_integer",
        "dense_3.weight_integer",
        "dense_3.bias_integer",
        "dense_4.weight_integer",
        "dense_4.bias_integer",
    ]
    for layer in layers:
        if name == layer:
            return True
    return False


def count_hawq_nonzero_weights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    for name, p in model.named_buffers():
        if check_layers(name):
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            layer_count_alive.update({name: nz_count})
            layer_count_total.update({name: total_params})
            nonzero += nz_count
            total += total_params
            logging.info(
                f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
            )
    logging.info(
        f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
    )
    return (
        nonzero,
        total,
        layer_count_alive,
        layer_count_total,
        (100 * (total - nonzero) / total),
    )


def calc_hawq_bops(model, input_data_precision=32):
    last_bit_width = input_data_precision
    alive, total, l_alive, l_total, pruned = count_hawq_nonzero_weights(model)
    b_w = model.dense_1.bias_bit if hasattr(model, "dense_1") else 32
    total_BOPS = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, QuantLinear):
            b_a = last_bit_width
            # b_w = module.quant_weight_bit_width #Dont think this is a property I can access sadly, going with precision as given set in model
            n = module.in_features
            m = module.out_features
            total = l_total[name + ".weight_integer"] + l_total[name + ".bias_integer"]
            alive = l_alive[name + ".weight_integer"] + l_alive[name + ".bias_integer"]
            p = 1 - ((total - alive) / total)  # fraction of layer remaining
            # assuming b_a is the output bitwidth of the last layer
            # module_BOPS = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
            module_BOPS = m * n * (p * b_a * b_w + b_a + b_w + math.log2(n))
            logging.info(
                "{} BOPS: {} = {}*{}({}*{}*{} + {} + {} + {})".format(
                    name, module_BOPS, m, n, p, b_a, b_w, b_a, b_w, math.log2(n)
                )
            )
            last_bit_width = b_w
            total_BOPS += module_BOPS
    logging.info("Total BOPS: {}".format(total_BOPS))
    return total_BOPS, pruned
