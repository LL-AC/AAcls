import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class resnet(nn.Module): 
    def __init__(self):
        super(resnet, self).__init__()

        self.img_features = resnet18(weights=ResNet18_Weights)
        self.img_features.fc = nn.Identity()

        self.classifier = nn.Sequential(nn.Linear(512, 128), 
                                        nn.ReLU(),
                                        nn.Linear(128, 3))
        
    def forward(self, x):
        x = self.img_features(x)
        
        x = self.classifier(x)
        return {'predicted': x}