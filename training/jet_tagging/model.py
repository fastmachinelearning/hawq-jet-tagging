import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune
from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantBnLinear


####################################################
# MLP (Floating-Point baseline)
####################################################
class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super(ThreeLayerMLP, self).__init__()

        self.dense_1 = nn.Linear(16, 64)
        self.act_1 = nn.ReLU()
        self.dense_2 = nn.Linear(64, 32)
        self.act_2 = nn.ReLU()
        self.dense_3 = nn.Linear(32, 32)
        self.act_3 = nn.ReLU()
        self.dense_4 = nn.Linear(32, 5)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.act_1(self.dense_1(x))
        x = self.act_2(self.dense_2(x))
        x = self.act_3(self.dense_3(x))
        return self.softmax(self.dense_4(x))


####################################################
# MLP with Batchnormalization 
####################################################
class ThreeLayer_BN(nn.Module):
    def __init__(self):
        super(ThreeLayer_BN, self).__init__()

        self.dense_1 = nn.Linear(16, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()

        self.dense_2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.dense_3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU()

        self.dense_4 = nn.Linear(32, 5)
        self.bn4 = nn.BatchNorm1d(5)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.act1(self.bn1(self.dense_1(x)))
        x = self.act2(self.bn2(self.dense_2(x)))
        x = self.act3(self.bn3(self.dense_3(x)))
        return self.softmax(self.bn4(self.dense_4(x)))


####################################################
# Base Model For Quantization
####################################################
class BaseModel(nn.Module):
    def __init__(self, weight_precision, bias_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision

    def init_dense(self, model, name):
        layer = getattr(model, name)
        quant_layer = QuantLinear(
            weight_bit=self.weight_precision, bias_bit=self.bias_precision
            # weight_bit=self.weight_precision, bias_bit=self.bias_precision, weight_percentile=95
        )
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)

    def init_bn_dense(self, model, fc_name, bn_name, bn_threshold, momentum):
        fc_layer = getattr(model, fc_name)
        bn_layer = getattr(model, bn_name)
        quant_layer = QuantBnLinear(
            weight_bit=self.weight_precision, bias_bit=self.bias_precision, fix_BN_threshold=bn_threshold
        )
        bn_layer.momentum = momentum
        quant_layer.set_param(fc_layer, bn_layer)
        setattr(self, fc_name, quant_layer)

    def prune(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                prune.random_unstructured(module, name="weight", amount=0.4)


class QThreeLayer(BaseModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer, self).__init__(weight_precision, bias_precision)

        self.quant_input = QuantAct(act_precision)

        self.init_dense(model, "dense_1")
        self.quant_act_1 = QuantAct(act_precision)

        self.init_dense(model, "dense_2")
        self.quant_act_2 = QuantAct(act_precision)

        self.init_dense(model, "dense_3")
        self.quant_act_3 = QuantAct(act_precision)

        self.init_dense(model, "dense_4")
        self.quant_output = QuantAct(act_precision)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)
        # act_scaling_factor = torch.tensor([1.])

        x = self.dense_1(x, act_scaling_factor)
        # x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)
        x = self.act(x)
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)

        x = self.dense_2(x, act_scaling_factor)
        # x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)
        x = self.act(x)
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)

        x = self.dense_3(x, act_scaling_factor)
        # x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)
        x = self.act(x)
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)

        x = self.dense_4(x, act_scaling_factor)
        # x, act_scaling_factor = self.quant_output(x, act_scaling_factor)
        x = self.softmax(x)
        return x


####################################################
# MLP for Coral Calculations (returns activations)
####################################################
class QThreeLayer_Coral(BaseModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer_Coral, self).__init__(weight_precision, bias_precision)

        self.quant_input = QuantAct(act_precision)

        self.init_dense(model, "dense_1")
        self.quant_act_1 = QuantAct(act_precision)

        self.init_dense(model, "dense_2")
        self.quant_act_2 = QuantAct(act_precision)

        self.init_dense(model, "dense_3")
        self.quant_act_3 = QuantAct(act_precision)

        self.init_dense(model, "dense_4")
        self.quant_output = QuantAct(act_precision)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        a1 = self.dense_1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)
        x = self.act(a1)

        a2 = self.dense_2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)
        x = self.act(a2)

        a3 = self.dense_3(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)
        x = self.act(a3)

        a4 = self.dense_4(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_output(a4, act_scaling_factor)
        x = self.softmax(x)

        return x, (a1, a2, a3, a4)


class QThreeLayer_BN(QThreeLayer):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer_BN, self).__init__(model, weight_precision, bias_precision)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(5)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x = self.dense_1(x, act_scaling_factor)
        x = self.act(self.bn1(x))
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)

        x = self.dense_2(x, act_scaling_factor)
        x = self.act(self.bn2(x))
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)

        x = self.dense_3(x, act_scaling_factor)
        x = self.act(self.bn3(x))
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)

        x = self.dense_4(x, act_scaling_factor)
        x = self.softmax(self.bn4(x))
        return x

####################################################
# Quantized MLP with Batchnorm Folding 
####################################################
class QThreeLayer_BNFold(BaseModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6, fix_bn_threshold=1, momentum=0.9):
        super(QThreeLayer_BNFold, self).__init__(weight_precision, bias_precision)

        self.weight_precision = weight_precision
        self.bias_precision = bias_precision

        self.quant_input = QuantAct(act_precision)

        self.init_bn_dense(model, "dense_1", "bn1", fix_bn_threshold, momentum)
        self.quant_act_1 = QuantAct(act_precision)

        self.init_bn_dense(model, "dense_2", "bn2", fix_bn_threshold, momentum)
        self.quant_act_2 = QuantAct(act_precision)

        self.init_bn_dense(model, "dense_3", "bn3", fix_bn_threshold, momentum)
        self.quant_act_3 = QuantAct(act_precision)

        self.init_bn_dense(model, "dense_4", "bn4", fix_bn_threshold, momentum)
        self.quant_output = QuantAct(act_precision)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # def forward(self, x):
    #     x, act_scaling_factor = self.quant_input(x)

    #     x, weight_scaling_factor = self.dense_1(x, act_scaling_factor)
    #     x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor, weight_scaling_factor)
    #     x = self.act(x)

    #     x, weight_scaling_factor = self.dense_2(x, act_scaling_factor)
    #     x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor, weight_scaling_factor)
    #     x = self.act(x)

    #     x, weight_scaling_factor = self.dense_3(x, act_scaling_factor)
    #     x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor, weight_scaling_factor)
    #     x = self.act(x)

    #     x, weight_scaling_factor = self.dense_4(x, act_scaling_factor)
    #     x, act_scaling_factor = self.quant_output(x, act_scaling_factor, weight_scaling_factor)
    #     return self.softmax(x)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x = self.dense_1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)
        x = self.act(x)

        x = self.dense_2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)
        x = self.act(x)

        x = self.dense_3(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)
        x = self.act(x)

        x = self.dense_4(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_output(x, act_scaling_factor)
        return self.softmax(x)

####################################################
# Quantized MLP with Dropout 
####################################################
class QThreeLayer_Dropout(QThreeLayer):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer_BN_Dropout, self).__init__(model, weight_precision, bias_precision)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x = self.dense_1(x, act_scaling_factor)
        x = self.act(self.dropout(x))
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)

        x = self.dense_2(x, act_scaling_factor)
        x = self.act(self.dropout(x))
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)

        x = self.dense_3(x, act_scaling_factor)
        x = self.act(self.dropout(x))
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)

        x = self.softmax(self.dense_4(x, act_scaling_factor))
        return x

####################################################
# Quantized MLP with Batchnorm & Dropout 
####################################################
class QThreeLayer_BN_Dropout(BaseModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer_BN_Dropout, self).__init__(weight_precision, bias_precision)

        self.weight_precision = weight_precision
        self.bias_precision = bias_precision

        self.quant_input = QuantAct(act_precision)

        self.init_dense(model, "dense_1")
        self.bn1 = nn.BatchNorm1d(64)
        self.quant_act_1 = QuantAct(act_precision)

        self.init_dense(model, "dense_2")
        self.bn2 = nn.BatchNorm1d(32)
        self.quant_act_2 = QuantAct(act_precision)

        self.init_dense(model, "dense_3")
        self.bn3 = nn.BatchNorm1d(32)
        self.quant_act_3 = QuantAct(act_precision)

        self.dropout = nn.Dropout(p=0.2)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x = self.bn1(self.dense_1(x, act_scaling_factor))
        x = self.act(self.dropout(x))
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)

        x = self.bn2(self.dense_2(x, act_scaling_factor))
        x = self.act(self.dropout(x))
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)

        x = self.bn3(self.dense_3(x, act_scaling_factor))
        x = self.act(self.dropout(x))
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)

        x = self.softmax(self.dense_4(x, act_scaling_factor))
        return x


####################################################
# Helper functions 
####################################################
def get_model(args):
    if args.batch_norm or args.batch_norm_fold:
        return ThreeLayer_BN()
    return ThreeLayerMLP()

def get_quantized_model(args, model=None):
    if model == None:
        model = get_model(args)
    if args.dropout and args.batch_norm:
        return QThreeLayer_BN_Dropout(model)
    if args.batch_norm:
        return QThreeLayer_BN(model)
    if args.batch_norm_fold:
        return QThreeLayer_BNFold(model)
    if args.dropout:
        return QThreeLayer_Dropout()
    return QThreeLayer(model)
