import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl


class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_shape, 72)
        self.bn1 = nn.BatchNorm1d(72)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(72, 72)
        self.bn2 = nn.BatchNorm1d(72)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(72, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(8, 72)
        self.bn4 = nn.BatchNorm1d(72)
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(72, 72)
        self.bn5 = nn.BatchNorm1d(72)
        self.relu5 = nn.ReLU()
        
        self.fc6 = nn.Linear(72, input_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.fc6(x)
        return x


####################################################
# AD08 PyTorch Lightning  
####################################################
class AD08(pl.LightningModule):
  def __init__(self, input_shape=64, precision=[], lr=1e-3) -> None:
    super().__init__()

    self.lr = lr
    self.loss = nn.MSELoss()
    self.model = Model(input_shape)
    self.warmup_epochs = 0
    self.weight_decay = 1e-4
    self.lr_decay = 0.99
    self.train_step_loss = []
    self.validation_step_loss = []

    if len(precision) != 0:
       print('Loading quantized model')
       self.model = Model(self.model, precision[0], precision[1], precision[2])

  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_idx):
    x, y, = batch 
    y_hat = self.model(x)
    loss = self.loss(y_hat, y)
    self.train_step_loss.append(loss.item())
    self.log("train_loss", loss, on_epoch=True, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y, = batch 
    y_hat = self.model(x)
    val_loss = self.loss(y_hat, y)
    self.validation_step_loss.append(val_loss.item())
    self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

  def test_step(self, batch, batch_idx):
    x, y, = batch 
    y_hat = self.model(x)
    test_loss = self.loss(y_hat, y)
    
    self.test_step_loss.append(test_loss)
    self.log('test_loss', test_loss)

  def on_train_epoch_start(self) -> None:
     self.train_step_loss.clear()
     return super().on_train_epoch_start()

  def on_validation_epoch_start(self) -> None:
     self.validation_step_loss.clear()
     return super().on_validation_epoch_start()
  
  def on_test_end(self) -> None:
      self.test_step_loss.clear()  # free memory
      return super().on_test_end()
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
    return optimizer


if __name__ == '__main__':
    x = torch.randn([2, 64])
    
    model = AD08()
    print(model(x))
