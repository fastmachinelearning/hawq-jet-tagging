import sys 
sys.path.append('../..')
import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import pytorch_lightning as pl
from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantBnLinear, QuantConv2d


class WarmUpLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        last_epoch: int = -1,
    ):
        """Scheduler for learning rate warmup.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer, e.g. SGD.
        total_iters: int
            Number of iterations for warmup Learning rate phase.
        last_epoch: int
            The index of last epoch. Default: -1
        """
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return current learning rate."""
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


####################################################
# Base Quantized Model 
####################################################
class BaseQModel(nn.Module):
    def __init__(self, weight_precision, bias_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision

    def init_dense(self, model, name, w_b):
        layer = getattr(model, name)
        quant_layer = QuantLinear(
            weight_bit=w_b, bias_bit=w_b
        )
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)
    
    def init_conv2d(self, model, name, w_b):
        layer = getattr(model, name)
        quant_layer = QuantConv2d(
            weight_bit=w_b, bias_bit=w_b
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
    
    def encodeBitwidths(self, precision, layers):
        self.map_bitwidth = {}
        for bitwidth, layer in zip(precision, layers):
            self.map_bitwidth[layer] = bitwidth


####################################################
# ResNet
####################################################
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv4_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv6_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print(out.shape)
        # print('__POINT_1__')

        residual = out  # Store the residual
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual  # Add the residual
        out = self.relu(out)

        # print(out.shape)
        # print('__POINT_2__')

        residual = out  # Store the residual
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)

        out = out + self.conv4_1(residual)  # Add the residual
        out = self.relu(out)

        # print(out.shape)
        # print('__POINT_3__')

        residual = out
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.bn7(out)

        out = out + self.conv6_1(residual)  # Add the residual
        out = self.relu(out)

        # print(out.shape)
        # print('__POINT_4__')

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


####################################################
# Quantized ResNet Model 
####################################################
class QResNet(BaseQModel):
    def __init__(self, model, weight_precision, bias_precision, act_precision=8) -> None:
        super().__init__(weight_precision, bias_precision)

        self.quant_input = QuantAct(act_precision)

        self.init_conv2d(model, "conv1")
        self.init_conv2d(model, "conv2")
        self.init_conv2d(model, "conv3")
        self.init_conv2d(model, "conv4")
        self.init_conv2d(model, "conv4_1")
        self.init_conv2d(model, "conv5")
        self.init_conv2d(model, "conv6")
        self.init_conv2d(model, "conv6_1")
        self.init_conv2d(model, "conv7")

        self.quant_act_1 = QuantAct(act_precision)
        self.quant_act_2 = QuantAct(act_precision)
        self.quant_act_3 = QuantAct(act_precision)
        self.quant_act_4 = QuantAct(act_precision)
        self.quant_act_5 = QuantAct(act_precision)
        self.quant_act_6 = QuantAct(act_precision)
        self.quant_output = QuantAct(act_precision)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)

        self.init_dense(model, "fc")
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)

    def forward(self, x):
        out, p_sf = self.quant_input(x)
        out, w_sf = self.conv1(out, p_sf)
        out = self.bn1(out)
        out = self.relu(out)
        out, p_sf = self.quant_act_1(out, p_sf, w_sf)

        # print(out.shape)
        # print('__POINT_1__')

        residual = out  # Store the residual
        out, w_sf = self.conv2(out, p_sf)
        out = self.bn2(out)
        out = self.relu(out)
        out, p_sf = self.quant_act_2(out, p_sf, w_sf)
        out, w_sf = self.conv3(out, p_sf)
        out = self.bn3(out)

        out += residual  # Add the residual
        out = self.relu(out)
        out, p_sf = self.quant_act_3(out, p_sf)

        # print(out.shape)
        # print('__POINT_2__')

        residual = out  # Store the residual
        out, w_sf = self.conv4(out, p_sf)
        out = self.bn4(out)
        out = self.relu(out)
        out, p_sf = self.quant_act_3(out, p_sf, w_sf)
        out, w_sf = self.conv5(out, p_sf)
        out = self.bn5(out)

        residual, w_sf = self.conv4_1(residual, p_sf)
        out = out + residual  # Add the residual
        out = self.relu(out)
        out, p_sf = self.quant_act_4(out, p_sf, w_sf)

        # print(out.shape)
        # print('__POINT_3__')

        residual = out
        out, w_sf = self.conv6(out, p_sf)
        out = self.bn6(out)
        out = self.relu(out)
        out, p_sf = self.quant_act_5(out, p_sf, w_sf)
        out, w_sf = self.conv7(out, p_sf)
        out = self.bn7(out)

        residual, w_sf = self.conv6_1(residual, p_sf)
        out = out + residual  # Add the residual
        out = self.relu(out)
        out, p_sf = self.quant_act_6(out, p_sf, w_sf)

        # print(out.shape)
        # print('__POINT_4__')

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out, p_sf)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


####################################################
# ResNetBlock
####################################################
class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                stride=stride,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )

    def forward(self, inputs):
        x = self.block(inputs)
        y = self.residual(inputs)
        return F.relu(x + y)


####################################################
# MLPerf Tiny 1.0 RN08
####################################################
class MLPerfTinyRN08(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(1,1), stride=(1,1), padding='same')
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 4, kernel_size=(4,4), stride=(1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(4)

        self.conv3 = nn.Conv2d(4, 32, kernel_size=(4,4), stride=(1,1), padding='same')
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=(4,4), stride=(4,4), padding='valid')
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=(4,4), stride=(1,1), padding='same')
        self.bn5 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(2048, 10)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # print('ConvBnBlock1:', x.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        # print('ConvBnBlock2:', x.shape)
        x = self.relu(self.bn3(self.conv3(x)))
        # print('ConvBnBlock3:', x.shape)
        x = self.relu(self.bn4(self.conv4(x)))
        # print('ConvBnBlock4:', x.shape)
        x = self.relu(self.bn5(self.conv5(x)))
        # print('ConvBnBlock5:', x.shape)
        x = torch.flatten(x, 1)
        x = self.softmax(self.fc(x))
        return x

####################################################
# Quantized MLPerf Tiny 1.0 RN08
####################################################
class QMLPerfTinyRN08(BaseQModel):
    def __init__(self, model, precision) -> None:
        super().__init__(weight_precision=4, bias_precision=4)
        self.layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc']
        self.encodeBitwidths(precision=precision, layers=self.layers)

        self.quant_input = QuantAct(16)
        self.init_conv2d(model, "conv1", self.map_bitwidth['conv1'])
        self.bn1 = nn.BatchNorm2d(32)
        self.q_relu_1 = QuantAct(self.map_bitwidth['conv1']+3)
        
        self.init_conv2d(model, "conv2", self.map_bitwidth['conv2'])
        self.bn2 = nn.BatchNorm2d(4)
        self.q_relu_2 = QuantAct(self.map_bitwidth['conv2']+3)
        
        self.init_conv2d(model, "conv3", self.map_bitwidth['conv3'])
        self.bn3 = nn.BatchNorm2d(32)
        self.q_relu_3 = QuantAct(self.map_bitwidth['conv3']+3)

        self.init_conv2d(model, "conv4", self.map_bitwidth['conv4'])
        self.bn4 = nn.BatchNorm2d(32)
        self.q_relu_4 = QuantAct(self.map_bitwidth['conv4']+3)

        self.init_conv2d(model, "conv5", self.map_bitwidth['conv5'])
        self.bn5 = nn.BatchNorm2d(32)
        self.q_relu_5 = QuantAct(self.map_bitwidth['conv5']+3)

        self.init_dense(model, "fc", self.map_bitwidth['fc'])
        self.q_relu_6 = QuantAct(self.map_bitwidth['fc']+3)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x, p_sf = self.quant_input(x)

        x, w_sf = self.conv1(x, p_sf)
        x = self.relu(self.bn1(x))
        x, p_sf = self.q_relu_1(x, p_sf, w_sf)

        x, w_sf = self.conv2(x, p_sf)
        x = self.relu(self.bn2(x))
        x, p_sf = self.q_relu_2(x, p_sf, w_sf)

        x, w_sf = self.conv3(x, p_sf)
        x = self.relu(self.bn3(x))
        x, p_sf = self.q_relu_3(x, p_sf, w_sf)

        x, w_sf = self.conv4(x, p_sf)
        x = self.relu(self.bn4(x))
        x, p_sf = self.q_relu_4(x, p_sf, w_sf)

        x, w_sf = self.conv5(x, p_sf)
        x = self.relu(self.bn5(x))
        x, p_sf = self.q_relu_5(x, p_sf, w_sf)

        x = torch.flatten(x, 1)
        x = self.fc(x, p_sf)
        x = self.softmax(x)
        return x

####################################################
# MLCommons ResNet 
####################################################
class Resnet8v1EEMBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
        )

        self.first_stack = ResNetBlock(in_channels=16, out_channels=16, stride=1)
        self.second_stack = ResNetBlock(in_channels=16, out_channels=32, stride=2)
        self.third_stack = ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.first_stack(x)
        x = self.second_stack(x)
        x = self.third_stack(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


####################################################
# RN08 PyTorch Lightning  
####################################################
class RN08(pl.LightningModule):
  def __init__(self, precision=[], lr=1e-3, quantize=False) -> None:
    super().__init__()

    self.lr = lr
    self.loss = nn.CrossEntropyLoss()
    # self.model = ResNet()
    self.model = MLPerfTinyRN08()
    if quantize:
        print('Loading quantized model')
        self.model = QMLPerfTinyRN08(self.model, precision)
    self.warmup_epochs = 0
    self.weight_decay = 1e-4
    self.lr_decay = 0.99
    self.validation_step_acc = []
    self.test_step_acc = []
    self.test_step_loss = []

  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_idx):
    x, y, = batch 
    y_hat = self.model(x)
    loss = self.loss(y_hat, y)
    self.log("train_loss", loss, on_epoch=True, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y, = batch 
    y_hat = self.model(x)
    val_loss = self.loss(y_hat, y)
    val_acc = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
    self.validation_step_acc.append(val_acc)
    self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

  def test_step(self, batch, batch_idx):
    x, y, = batch 
    y_hat = self.model(x)
    test_loss = self.loss(y_hat, y)
    test_acc = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
    
    self.test_step_loss.append(test_loss)
    self.test_step_acc.append(test_acc)
    self.log('test_loss', test_loss)
    self.log('test_acc', test_acc)
  
  def on_train_epoch_start(self) -> None:
      if self.current_epoch > self.warmup_epochs:
            self.train_scheduler.step()
      return super().on_train_epoch_start()

  def on_train_epoch_end(self) -> None:
     if self.current_epoch <= self.warmup_epochs:
         self.warmup_scheduler.step()
     else:
        self.train_scheduler.step(self.current_epoch)
     return super().on_train_epoch_end()  
  
  def on_validation_epoch_end(self) -> None:
      self.log('val_acc_test', np.array(self.validation_step_acc).mean(), on_epoch=True, prog_bar=True)
      self.validation_step_acc.clear()  # free memory
      return super().on_validation_epoch_end()
  
  def on_test_end(self) -> None:
#       test_loss = np.array(self.test_step_loss).mean()
#       self.log.experiment('test_loss_FINAL', test_loss)
#       test_acc = np.array(self.test_step_acc).mean()
#       self.log('test_acc', test_acc)
      self.test_step_loss.clear()  # free memory
      self.test_step_acc.clear()  # free memory
      return super().on_test_end()
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
    self.train_scheduler = LambdaLR(
        optimizer=optimizer, lr_lambda=lambda epoch: self.lr_decay**epoch
    )
    self.warmup_scheduler = WarmUpLR(optimizer=optimizer, total_iters=self.warmup_epochs)
    return optimizer


if __name__ == '__main__':
    x = torch.randn([1, 3, 32, 32])
    
    model = ResNet()
    print(count_parameters(model))
    print(model(x))

    # model = QResNet(model, 16, 16)
    # print(count_parameters(model))
    # print(model(x))
    