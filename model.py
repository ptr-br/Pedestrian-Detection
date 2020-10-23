# file that  builds the classification model 

#imports
from torchvision import datasets, transforms, models
from torch import nn
import torch

def getModel(path="./saved_models/pedestrianClassifier5Epochsiou045-2020-10-20-11-15-47.pt"):
    '''
    getModel creates pytorch model and sets trained weights 
    
    params:
            path: path to the saved weights
    
    return:
            pytorch model for pedestrian classification
    
    '''
    # get model from library
    model = models.resnet50(pretrained=True)
    # remove original output layer and replace it with 2 dimensinal layer 
    number_input_features  = model.fc.in_features
    num_classes =2
    model.fc = nn.Linear(number_input_features, num_classes)
    # load model trained in second notebook
    model.load_state_dict(torch.load('./saved_models/pedestrianClassifier12Epochs-iou45-2020-10-21-13-45-26.pt'))
    
    return model