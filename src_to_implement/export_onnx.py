import torch as t
from trainer import Trainer
import sys
import torchvision as tv
from model import ResNet,ResidualLayer
epoch = int(sys.argv[1])
#TODO: Enter your model here

crit = t.nn.BCELoss()
model = ResNet(ResidualLayer,2)

trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
