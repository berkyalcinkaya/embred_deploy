
# model weights: model_class" 
from .models import *

model_mapping = {"Wnet_weightedLoss_embSplits":[WNet, {"num_classes":13}],
 "SimpleNet_weightedLoss_embSplits":[SimpleNet3D, {"num_classes":13}],
 "CustomResNet50Unfrozen_CE_balanced_embSplits": [CustomResNet50, {"num_classes":14, "num_dense_layers": 0, "dense_neurons": 128, "freeze_": False}],
 "UnfrozenResNet50_smaller": [CustomResNet50, {"num_classes":14, "num_dense_layers": 0, "dense_neurons": 64, "freeze_": False}],
 "UnfrozenResNet50_1layer64": [CustomResNet50, {"num_classes":14, "num_dense_layers": 1, "dense_neurons": 64, "freeze_": False}], 
 "New-ResNet50-Unfreeze-CE-embSplits-overUnderSampleMedian-lessregularized-nodropout-3layer256,128,64": [CustomResNet50, {"num_classes":14, "num_dense_layers": 3, "dense_neurons": [256, 128, 64], "freeze_": False}]
 }