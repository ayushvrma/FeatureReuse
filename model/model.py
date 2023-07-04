import torch
import torch.nn as nn
import torchvision.models as models

class RetinaNet(nn.Module):
    def __init__(self, num_classes):
        super(RetinaNet, self).__init__()
        
        # Load the pre-trained VGG16 model
        self.pretrained_model = models.vgg16(pretrained=True)
        
        # Freeze the weights of the pre-trained layers
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer with a new one
        num_features = self.pretrained_model.classifier[6].in_features
        self.pretrained_model.classifier[6] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        return x
