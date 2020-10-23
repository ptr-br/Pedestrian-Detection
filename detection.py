# this file contains a transform and a detect function
# it is used inthe notebooks to detect a pedestrian in a given image


from helpers import sliding_window, image_pyramid, display_bounding_box
from torchvision import datasets, transforms, models
from imutils.object_detection import non_max_suppression
from torch import nn
import torch
import os
import numpy as np

def prediction_transforms(roi):
    '''
    transorms the input image to a format that can be read by the network
    
    params: 
            roi: region of interest (PIL.Image) that gets transformed into tensor 
            
    return:
            roi: roi in tensor format
    '''
    
    
    pred_transform = transforms.Compose([        
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    roi = pred_transform(roi)
        
    return roi

def detect_pedestrian(image, model,slidingWindow_parameters={'height':int(275),'width':int(100),'step_w':50,'step_h':50},imagePyramid_parameters=(224,224)):
    '''
    detects pedestrians ifrom a given input image 
    
    params:
            image: PIL Input image in which pedestrians should be detected
            model: pytorch model used for prediction
            slidingWindow_parameters: Dictionary that contains information about the slliding window (height,width,step size in vertical and horizontal direction)
            imagePyramid_parameters: Tuple that contains the smallest size that the image should be scaled to
    
    return:
            picks: bounding box coordinates from the prediction
    
    '''

    listOfBoundningBoxes = []
    listOfProbs =[]
    
    locs = []
    rois = []
    
    ImgWidth,imgHeight = image.size
    
    # initialize image pyramid
    imgPyramide = image_pyramid(image,minSize=imagePyramid_parameters)
    
    for img in imgPyramide:
        
        # get current scale to rescale roi later 
        scale = ImgWidth/float(img.size[0])
        
        # create sliding window generator 
        sliding = sliding_window(img,
                                 window_size=(slidingWindow_parameters['width'],slidingWindow_parameters['height']),
                                 step_w=int(slidingWindow_parameters['step_w']/scale),
                                 step_h=int(slidingWindow_parameters['step_h']/scale))  
        
        
        # initialize sliding window
        for slide_img in sliding:
            
            windowBox = slide_img[0]
            windowBox= tuple(int(x*scale) for x in windowBox)
            
            locs.append(windowBox)
       
            
            # prepare region of interest for input into the classifier
            roi =slide_img[1].resize((224,int(224*2.5)))
            roi = prediction_transforms(roi)
            rois.append(roi)    
    


    
    # classify the roi
    model.eval()
    with torch.no_grad():
        rois =torch.stack(rois, dim=0)
        
         
        
        predLoader = torch.utils.data.DataLoader(rois,batch_size=8)
        sm = torch.nn.Softmax(dim=1)
        outputs = []
        
        # split rois to prevent memory overload
        for inputs in predLoader:
            # use model to classify rois
            outputs.append(model(inputs))
        outputs =torch.cat(outputs, dim=0)
        
        _, preds = torch.max(outputs.data, 1)
        probs= sm(outputs)
        
        
        # get list of indexes that conatain a pedestrian
        indexes = preds.numpy().nonzero()
       
        for index in indexes[0]:
            print(f"Detected pedestrian at index {index}.")
            listOfBoundningBoxes.append(locs[index])
            listOfProbs.append(probs[index][1])
    

        
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[xmin, ymin, xmax, ymax] for (xmin, ymin, xmax, ymax) in listOfBoundningBoxes])
    picks = non_max_suppression(rects, probs=listOfProbs, overlapThresh=0.1)
        
    return picks
                    