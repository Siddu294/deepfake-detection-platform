The supported model architecture is defined in resnet_detector.py and matches
train_advanced.py's ResNet-50 head (Dropout(0.4) + Linear(2048, 2)).
The class order is fake=0, real=1. Old custom-CNN checkpoints are not compatible.
