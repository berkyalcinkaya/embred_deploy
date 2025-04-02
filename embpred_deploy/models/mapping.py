
# model weights: model_class" 
from .models import *

mapping = {"Wnet_weightedLoss_embSplits":[WNet, {"num_classes":13}],
 "SimpleNet_weightedLoss_embSplits":[SimpleNet3D, {"num_classes":13}],
 "CustomResNet50Unfrozen_CE_balanced_embSplits": [CustomResNet50, {"num_classes":14, "num_dense_layers": 0, "dense_neurons": 128, "freeze_": False}],
 }