import torch.nn as nn
import torch.nn.utils.prune as prune
from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear

from .three_layer import get_model


class BaseModel(nn.Module):
    def __init__(self, weight_precision, bias_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision

    def init_dense(self, model, name):
        layer = getattr(model, name)
        quant_layer = QuantLinear(
            weight_bit=self.weight_precision, bias_bit=self.bias_precision
        )
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)

    def prune(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                prune.random_unstructured(module, name="weight", amount=0.9)


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

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x = self.act(self.dense_1(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)

        x = self.act(self.dense_2(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)

        x = self.act(self.dense_3(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)

        x = self.softmax(self.dense_4(x, act_scaling_factor))
        return x


class QThreeLayer_BN(BaseModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer_BN, self).__init__(weight_precision, bias_precision)

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

        self.init_dense(model, "dense_4")

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

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

        x = self.softmax(self.dense_4(x, act_scaling_factor))
        return x


class QThreeLayer_Dropout(BaseModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer_BN_Dropout, self).__init__(weight_precision, bias_precision)

        self.weight_precision = weight_precision
        self.bias_precision = bias_precision

        self.quant_input = QuantAct(act_precision)

        self.init_dense(model, "dense_1")
        self.quant_act_1 = QuantAct(act_precision)

        self.init_dense(model, "dense_2")
        self.quant_act_2 = QuantAct(act_precision)

        self.init_dense(model, "dense_3")
        self.quant_act_3 = QuantAct(act_precision)

        self.dropout = nn.Dropout(p=0.2)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

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


def get_quantized_model(args, model=None):
    if model == None:
        model = get_model(args)
    if args.dropout and args.batch_norm:
        return QThreeLayer_BN_Dropout(model)
    if args.batch_norm:
        return QThreeLayer_BN(model)
    if args.dropout:
        return QThreeLayer_Dropout()
    return QThreeLayer(model)
