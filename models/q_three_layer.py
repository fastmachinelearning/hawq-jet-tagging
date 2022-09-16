import torch.nn as nn
from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear

from .three_layer import three_layer_mlp


class QThreeLayer(nn.Module):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer, self).__init__()

        self.quant_input = QuantAct(act_precision)

        layer = getattr(model, "dense_1")
        quant_layer = QuantLinear(weight_bit=weight_precision, bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, "dense_1", quant_layer)
        self.quant_act_1 = QuantAct(act_precision)

        layer = getattr(model, "dense_2")
        quant_layer = QuantLinear(weight_bit=weight_precision, bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, "dense_2", quant_layer)
        self.quant_act_2 = QuantAct(act_precision)

        layer = getattr(model, "dense_3")
        quant_layer = QuantLinear(weight_bit=weight_precision, bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, "dense_3", quant_layer)
        self.quant_act_3 = QuantAct(act_precision)

        layer = getattr(model, "dense_4")
        quant_layer = QuantLinear(weight_bit=weight_precision, bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, "dense_4", quant_layer)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x = self.act(self.dense_1(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)

        x = self.act(self.dense_2(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)

        x = self.act(self.dense_3(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_3(
            x,
            act_scaling_factor,
        )

        x = self.softmax(self.dense_4(x, act_scaling_factor))
        return x


def q_three_layer(model=None):
    if model == None:
        model = three_layer_mlp()
    return QThreeLayer(model)
