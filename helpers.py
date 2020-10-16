#imports 
import imutils
import os
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def display_bounding_box(image,box_coordinates, display=True, resize=(0,0),cropBox=None):
    '''
    Display_bounding_box is a function that draws the corresponding bounding boxes into the images
 
    params: 
             image: Image file or patht to the image file
             box_coordinates: Coordinadtes of the bounding boxes inside the image (np-array)
             display: Whether the image and the corrsponding bounding box should be displayed
             resize: Tuple values to resize image to given input width
    '''
    # if path is provided get image, else image is already given
    if isinstance(image, str):
        image = cv2.imread(image)
        # switch red and blue color channels 
        # -> by default OpenCV assumes BLUE comes first, not RED as in many images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
    elif isinstance(image, Image.Image):
        image = np.array(image)
        
    # resize the input image as well as the corresponding bounding boxes to a fixed width
    if resize[0]!=0:
        originalShape = image.shape
        #image = imutils.resize(image, width=resize[0])  <-- keeps original image ratio 
        image = cv2.resize(image, resize, interpolation = cv2.INTER_AREA)
        newShape = image.shape
        new_coord = lambda bc,oS,nS: bc*nS[0]/oS[0]
        box_coordinates = new_coord(bc=box_coordinates,oS=originalShape,nS=newShape)
                               
    # loop over the detected persons, mark the image where each person is found
    for (xmin,ymin,xmax,ymax) in box_coordinates:
        # draw a rectangle around each detected person
        cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,0),2) 
        
    if cropBox != None:
        # draw a blue rectangle around the box that should be cropped out
        xmin, ymin, xmax, ymax = cropBox
        cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),2) 
        

    if display==True:                   
        # plot the image
        fig = plt.figure(figsize=(9,9))
        plt.imshow(image)
    else:
        return image
    

def sliding_window(image, step_w,step_h, window_size):
    '''
    Implemetation of a sliding window

    prarms
        image      : Input image from which regions are created
        step       : Step size that indicates how many pixel window is shifted per iteration (in x and y)
        window_size: Defines the size of the sliding window (width and height)
    
    return
        Generator functions yields back the window 
    '''
    
    for y in range(0, image.shape[0] - window_size[1], step_h):      
        for x in range(0, image.shape[1] - window_size[0], step_w):            
            yield (x, y, image[y:y+window_size[1] , x:x+window_size[0] ])   # yield back current window
            
            
            
def image_pyramid(image, scale=1.5, minSize=(224,224)):
   
    '''
    Implemetation of a image pyramid
    
    prarms
        image      : Input image from which pyramid is created
        scale      : Scale factor (controls how much the image is resized az each iteration)
        min_size   : Defines the minimum size until the pyramide is to small 
    
    return
        Generator functions yields back the window 
    ''' 
    
    # the lowest pyramide level is equal to the original image
    yield image
    # now downsample pyramid until minsize is reached 
    
    
    while True:      
        w = int(image.shape[1]/scale)      
        image = imutils.resize(image, width=w)       
        # if min size is reached
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
            
        yield image 
        
        
        
def listdir_nohidden(path):
    '''
    works like os listdir, but ignores hidden files that start with a "."
    '''
    list_dir_list= []
    for f in os.listdir(path):
        print(f)
        print(os.getcwd())
        if not f.startswith('.'):
            print('JF')
            list_dir_list.append(f)
        return list_dir_list
    
    
def check_neg_boundingBox(boundingBox_coordinates,x_upperlLeft,y_upperLeft,boxwidth,boxheight):    
    '''
    Check if the randomly crop of the image contains a pedestrian or not
 
    params: 
             boundingBox_coordinates: Coordinates of the persons in the image (list with entys [[xmin,ymin,xmax,ymax],[...]])
             x_upperlLeft: X-coordinate in the image 
             y_upperLeft: Y-coordinate in the image
             boxwidth: Denotes the width of the box that is cropped
             boxheight: Denotes the height of the box that is cropped
                
    return:
            Boolean that indicates whether the image crop contains a pedestrian or not 
                 
    '''   
    # check if randomly created bounding box intersects with pedestrian
    for box in boundingBox_coordinates:    
        # NOTE: [0]=xmin [1]=ymin [2]=xmax [3]=ymax
        
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box[0], x_upperlLeft)
        yA = max(box[1], y_upperLeft)
        xB = min(box[2], x_upperlLeft+boxwidth)
        yB = min(box[3], y_upperLeft+boxheight)
        # compute the area of intersection rectangle
        intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        if intersectionArea > 0:
            return False
        
    return True
            
        
        
        

        
        
        
        
        
        
        
        