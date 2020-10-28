# Pedestrian-Detection
## Introduction

This is the final project of the [Udacity Machine Learning Engineer Nanodegree Program](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t).


As final project i created a simple pedestrian detection system, that useses an image classifier and a sliding window to detect pedestrians in an given input image.
The dataset used in this project is from the [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/).
For the classifier, the resnet50 architecture is adapted (feature extraction on final fully conected layer).  

## Included in this repository
- Data Exploration and Generation.ipynb - get familiar with the data and generate classifier input images
- The Classifier.ipynb - setting up and traning of the classifier
- Pedestrian Detection Pipeline.ipynb - End to end usage of the created system on examples
- Benchmark Model and Evaluation.ipynb - Setup and evaluation of both models
- DataClass.py - Contains PennFudanDataset class that is useful to load the data and work with it 
- detection.py - Outsourced detect function of the produced model (for usage across notebooks)
- helpers.py   - useful helper fuctions (such as sliding window, image pyramid, display bounding boxes, etc.)
- model.py     - detection model with tranined weights

## Setting up the environment

### Dataset
The zip-file has to be downloaded manually. 
To unzip the files the first notebook can be used.


### Libraries
This project is developed in Python 3.6
You will need install some libraries in order to run the code.  
Libraries and respective version are:  

- jupyter 1.0.0
- opencv-python
- numpy 1.18.1 
- pandas 1.0.1
- pathlib2 2.3.5
- Pillow 7.0.0
- zipp 2.2.0
- matplotlib 3.1.3
- torch 1.4.0
- torchvision 0.5.0
- glob2 0.7
- natsort 7.0.1
- imutils 0.5.1
- split-folders 0.4.2
