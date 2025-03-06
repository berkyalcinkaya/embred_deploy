
# model weights: model_class" 
from .models import *

mapping = {"Wnet_weightedLoss_embSplits":[WNet, {"num_classes":13}],
 "SimpleNet_weightedLoss_embSplits":[SimpleNet3D, {"num_classes":13}]
 }