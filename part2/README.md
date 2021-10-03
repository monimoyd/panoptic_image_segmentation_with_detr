# Predicting Bounding Box for Enginering Materials Dataset
In this project I have Using Deep Learning Techinques to predict bounding boxes for Engineering materials

# I. Problem Statement

In this project I have Using Deep Learning Techinques to predict bounding boxes for Engineering materials 

The dataset is available in link below:

https://drive.google.com/file/d/1IsK268zLnXB2Qq0X2LgNDwBZRuVwvjRx/view?usp=sharing




Major Highlights:

-  Used Panoptic DETR is used to generated ground truth of stuff for images givens
-  Final dataset is combined with coco stuff dataset
-  Fine tuning is done on the final dataset



## Running the code

You can use download the code

git clone https://github.com/monimoyd/panoptic_image_segmentation_with_detr.git 

Jupyter Notebook link:

There are two notebooks (Becuase of large size sharing the colab link)

https://colab.research.google.com/drive/1pcyq_DZYr5OxonJseosyWq4p702hq3Yq?usp=sharing : Used for training 
https://colab.research.google.com/drive/11_CaOlQ_ON9KeNBaVNoY4hE3Gpi8eug8?usp=sharing : For prediction

The trained model can be found in link below:

https://drive.google.com/file/d/1SHgGDvWyT5kvoWzno-WIPORyLl4rKpO-/view?usp=sharing
 
## II. DETR and Loss functions Used

In this project I have used DETR a deep learning based transformer model to predict bounding boxes for Engineering Materials
Dataset. DETR which stands for DEtection TRansformer is developed by Facebook AI and probably the most effective Computer Vision
 Algorithm which is empowered with Object classification, Object Detection, Semantic Segmentation and many other Deep learning 
 tasks. It extends the Transformer based architecture by infusing more intelligence w.r.t adding many object queries to the
 Decoder Architecture.
 
 To train the algorithm and evaluate its performance, a loss function is defined which is a sum of negative log-likelihood loss for classification and a combination of L1 (absolute difference) and generalized Intersection over Union (IoU) losses for the bounding box coordinates. The loss is based on matching each ground truth object to a predicted object, for which the Hungarian algorithm is used.
 
 More details about DETR can be found in URL below:
 
 https://github.com/nkanungo/EVA6_DETR#readme

## III. Data Cleaning and Data Loading

The engineering material dataset has 48 categories of various engineering materials like cu_piping , adhesives etc.

### Structure of the dataset

.
├── class_name_1   
│   ├── class_details.txt   
│   ├── coco.json   
│   └── images   
│       ├── img_000.png   
│       ├── img_001.png   
├── class_name_2   
│   ├── class_details.txt   
│   ├── coco.json   
│   └── images   
│       ├── img_000.png   
│       ├── img_001.png   
.


Each image has annnotation for just the annotation for the given category. As there was no
stuff annotations present.

To add the stuff annotation, each of the image is predicted with DETR panoptic segmentation code having
resnet101 backbone which was pretrained with Coco dataset. The bounding boxes predicted by  DETR panoptic segmentation
 only those bounding boxes are considered for which confidence is more than 0.85.

The jupter notebook generate_ground_truth.ipynb is used to convert 

The bounding box predicted by DETR panoptic segmentation are normalized format (0-1) and it has Xcenter, Ycenter, width and height
but the coco dataset bounding box is stored in (Xleft, Yleft, width, height) which are actual pixel position. To convert
the bounding boxes predicted by DETR panoptic segmentation to COCO format the folowing formula is used:

Xleft = Image Width * ( Predicted Xcenter - 0.5 *  Predicted width)
Yleft = Image Height * ( Xcenter - 0.5 * Predicted height)


The binary mask predicted by DETR panoptic segmentation is converted to RLE format for each object in image using 
pycocotools mask.encode API.




Categories used for engineering dataset are as below:

0 - Misc Stuff (these are all the Coco thing object predicted by DETR panoptic segmentation code  
1-48 - Classes used for engineering materials  
49 - 63 - Coco stuff labels are hirerachially organized where some of categories are grouped together under super category. I have all the supercategories as classses these are :
    49. building  
    50. ceiling  
    51. floor
    52. food
    53. furniture
    54. ground
    55. plant
    56. rawmaterial
    57. sky
    58. solid
    59. structural
    60. textile
    61. wall
    62. water
    63. window


For each annotation json provided for each category an intermediate csv file with fields as below:

id - A unique annotation id  
image_id - A unique image id  
image_path - path where image is lying  
width - width of image  
height - height of image  
bbox - Bounding box  
segment_map - Segmentation map converted to RLE format  
source - Source from where record is generated. 1 for annotation file, 2 for predicted from DETR panoptic segmentation code, 3 for coco stuff dataset  
area - Area of bounding box  
segment_polygon - Polygon coordinates for segments  


As the engineering materials dataset does not  have all the stuff, so the coco validation stuff dataset is added for training and for this also an intermediate csv file is created

While generating csv files, Any of the image which is not RGB  (like RGBA or grey image) or images in WEBP format are ignored.

All the intermediate csv files are combined using into two jsons, custom-val.json and custom-train.json
using Jupyter notebook. While combining all the csv files , each image is given a unique id with format category name concated with image id.

## IV. Workflow
Workflow is explained using the diagram below:

![workflow](/part2/images/workflow.png) 


Here all class jsons and the stuff prediction using panoptic segmentation are combined to a corresponding class csv file.
In addition stuff annotation for coco validation dataset is also converted to coco_stuff.csv. All the csvs are combined into 
two annotation json files custom-train.json and custom-val.json. From the processing valid images from engineering materials
dataset 90% are randomly assigned to training and 10% to validation. However all the images from Coco validation dataset are
assigned to training only


## V. Training


As the combined image size is very large, it is split into multiple zip files and then unzipped using the structure below


/content/data/custom  
├ annotations/  # JSON annotations  
│  ├ annotations/custom_train.json  
│  └ annotations/custom_val.json  
├ train2017/    # training images  
└ val2017/      # validation images  



The DETR code used for training from repository https://github.com/woctezuma/detr.git and pretrained weights are loaded from 
URL https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth' 


Hyperparameters:

Number of queries: 30
Learning Rate: 1e-5
Number of classes: 64

As the classes provided were imbalanced, To balance it during the training, I used the following techniques

As the classes number of object classes were imbalanced, to solve this the following techniques are used:

1. Number of objects per class is limited to 250 but multiple datasets are created so that all classes can be trained. While training 
used all these processed datasets at different epochs
3. Coco Stuff annnotations are added from Training dataset
4. class weights are adjusted to give more weightage to minority classes and less weightage to majority classes. misc_stuff which has more than 10000 objects is given weightage of 0.1, any of the classes which has less than 100 objects is given class weightage of 2 and all others are given weightage of 1.

Training boxes training is done for 50 epcochs during that time loss is very low

## VI. Loss Functions used

### CE Loss:

This is the classification cross entropy loss between predicted class and actual class

### L1 Loss

This is the L1 regression loss between predicted and target bounding boxes

### GIOU Loss:
GIOU is a improved version of IoU loss
In IoU, where there is no intersection, IoU has no value and therefore no gradient. GIoU however, is always differentiable.
GIOU , which is formulated as follows:

GIoU=|A∩B||A∪B|−|C∖(A∪B)||C|=IoU−|C∖(A∪B)||C|
Where A and B are the prediction and ground truth bounding boxes. C is the smallest convex hull that encloses both A and B. 


## Bounding Box Loss:

Bounding Box Loss is a combination of L1 Loss and GIOU loss between predicted and target bounding boxes

## Total Loss:
Total Loss is combination of CE Loss, Bounding Box Loss and Mask Loss. During panoptic training, I used default loss weights during bounding box training 

bbox_loss_coef: 5
giou_loss_coef: 2
set_cost_class: 1



## VI.  Results:


### i.

#### Ground truth with bounding boxes:

![image1_g](/part2/images/image1_new_ground_truth.png)


#### Predicted bounding boxes:

![image1_p](/part2/images/image1_new_predicted.png)


### ii.

#### Ground truth with bounding boxes:

![image2_g](/part2/images/image2_new_ground_truth.png)


#### Predicted bounding boxes:

![image2_p](/part2/images/image2_new_predicted.png)



### iii.

#### Ground truth with bounding boxes:

![image3_g](/part2/images/image3_new_ground_truth.png)


#### Predicted bounding boxes:

![image3_p](/part2/images/image3_new_predicted.png)



### iv.

#### Ground truth with bounding boxes:

![image4_g](/part2/images/image4_new_ground_truth.png)


#### Predicted bounding boxes:

![image4_g](/part2/images/image4_new_predicted.png)


### v.

#### Ground truth with bounding boxes:

![image4_g](/part2/images/image5_new_ground_truth.png)


#### Predicted bounding boxes:

![image5_g](/part2/images/image5_new_predicted.png)



### vi.

#### Ground truth with bounding boxes:

![image6_g](/part2/images/image6_new_ground_truth.png)


#### Predicted bounding boxes:

![image6_g](/part2/images/image6_new_predicted.png)


### vii.

#### Ground truth with bounding boxes:

![image7_g](/part2/images/image7_new_ground_truth.png)


#### Predicted bounding boxes:

![image8_p](/part2/images/image7_new_predicted.png)


### viii.

#### Ground truth with bounding boxes:

![image7_g](/part2/images/image8_new_ground_truth.png)


#### Predicted bounding boxes:

![image8_p](/part2/images/image8_new_predicted.png)


### ix.

#### Ground truth with bounding boxes:

![image7_g](/part2/images/image9_new_ground_truth.png)


#### Predicted bounding boxes:

![image8_p](/part2/images/image9_new_predicted.png)


### x.

#### Ground truth with bounding boxes:

![image7_g](/part2/images/image10_new_ground_truth.png)


#### Predicted bounding boxes:

![image8_p](/part2/images/image10_new_predicted.png)



## VII. Evalutation Metrics Used and Plots:

Various metrics used are:

### Loss:

This is the total loss including classification loss, bounding box loss and giou loss

### mAP : 
is Mean Average Precision. Its use is different in the field of Information Retrieval (Reference [1] [2] )and Multi-Class classification (Object Detection) settings. To calculate it for Object Detection, you calculate the average precision for each class in your data based on your model predictions. Average precision is related to the area under the precision-recall curve for a class. Then Taking the mean of these average individual-class-precision gives you the Mean Average Precision. 

### GIOU Loss:
GIOU is a improved version of IoU loss
In IoU, where there is no intersection, IoU has no value and therefore no gradient. GIoU however, is always differentiable.
GIOU , which is formulated as follows:

GIoU=|A∩B||A∪B|−|C∖(A∪B)||C|=IoU−|C∖(A∪B)||C|
Where A and B are the prediction and ground truth bounding boxes. C is the smallest convex hull that encloses both A and B. 


The following are plots 

### i. Loss and mAP Plots

![image9](/part2/images/plot1_new.png)


### ii. Loss CE, Loss Bounding Box, Loss GIOU Plots

![image12](/part2/images/plot2_new.png)



### iii. Class Error, Cardinality Error Unscaled plot Plot

![image12](/part2/images/plot3_new.png)



##  VIII. Issues faced

### i. Json marshalling issue

While converting to json from CSV, faced Exception that says marshalling can not be performed because int64 is used. This is
because when I loaded dataframe from csv file and converted to basic type it was numpy.int64 format which JSON
could not marshal. I need to convert the type explicitly to int to fix the issue 

### ii. Changin number of queries

I used number of queries as 30 specified it as a paramer in main.py. While inferring when I loaded the model and loaded
the weights from checkpoint, I was getting Runtime Exception of dimension mismatch as the model query embedding has dimension
(30, 256) but it was expecting dimension (100, 256)

To solve this, I explicitly changed the query embedding dimension using the following lines:

model.num_queries=30
model.query_embed = torch.nn.modules.sparse.Embedding(30, 256)

Also before training, I used the additional line below:

del checkpoint["model"]["query_embed.weight"]


### iii. Junk characters in segmentation

While checking the segmentation field in coco stuff annotation, I initially thought these are junk characters. After study
and discussions, I realized these are mask encoded in RLE format




##  IX. Conclusion


In this project I have learned how to prepare engineering materials dataset, right from image collection to annotation
using CVT tool. Next, I cleaned and prepared dataset by applying DETR panoptic segmentation code to get ground truth for 
stuff. Next, I did fine tuning using DETR model. This project gave me exposure to end to end Object detection.















