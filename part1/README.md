# Panotic Segmenation using DETR

In computer vision, Panoptic Segmentation is the task of breaking down the scene semantically so that each instance of a class, say a car 
or person, be labeled uniquely. It requires that an algorithm understand the difference between amorphous stuff like the sky
and ground and countable things like cars and persons.

Sample panoptic segmented image is as below:

![image3](/part1/panoptic_image3.png)

The process for generating pantopic segmentation is given below:

![segmentation_prcess](/part1/segmtation_process.png)


The following are answers to questions asked as part of capstone part1

## FROM WHERE DO WE TAKE THIS ENCODED IMAGE?

The original image has dimension of 3xHxW,
where 3 is number of channels corresponing to RGB, 
H is Height of Image
W is width of Image


A backbone network like RESNET50 is used to extract the features from image and output of backbone network is p x H/32 x W/32
p is generally used as 2048 which is very high.

Output from backbone network is again passed through 1x1 convolution layer to further reduce the number of channels to d so that
final encoded image feature dimension are d x H/32 x W/32, d is typically 256


## WHAT DO WE DO HERE?

The features extracted from the image of dimension are flattened to d x (H/32 x W/32) and  it is supplemented with a positional
embedding before passing it into a transformer encoder. The transformer decoder takes fixed number of learned embeddings, which we call object queries, and additionally attends to the encoder output and calculates attention score for each object embedding.

As each object query has dimension d and there are N object queries, total dimension of object query embedding is dxN

As there are N object queries and M head attention, the output attention score is of dimension:
N x M x H/32 x W/32


## WHERE IS THIS COMING FROM?

The previous steps generate attention mask. These attention masks are low resolution, so it needs to be transformed to the high resolution so that each mask corresponds to one pixel in image.  

For doing this a FPN style convolution network is used. FPN style convolution network concatenates attention maps from different heads
using ResNet5, followed by upsampling, followed by ResNet4 , followed by upsampling, followed by ResNet3 followed by upsampling, followed by ResNet2 and followed by convolution and finally generates map of dimension N x H/4 x W/4. The map is high resolution where each pixel contains binary logit belonging to the mask


## EXPLAIN THESE STEPS?

The steps are as below:

i. Convolution network (RESNET 50) is used to extract the features from the input image
ii. After object detection the model is frozen and used for panoptic segmentation

ii. The features are given to multihead transformer, whose decoder uses p object queries to generate low resolution attention maps. 
iii. FPN style convolution network is used to convert low resolution attention maps to high resolution mask

iv. All the high resolution masks are combined by the assigning each pixel value  the argmax corresponding to highest logits

v. The Ground truth of panoptic segmentation from image was generated previously by predicting segments from stuffs from Coco datasets using a pretrained model combining with our annonated segments 

vi. Loss is calculated between ground truth and the generated panoptic segmentation. Loss function is a combination of dice loss and focal loss. AdamW is used as optimizer. 



