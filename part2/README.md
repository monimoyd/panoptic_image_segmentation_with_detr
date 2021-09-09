# Predicting Bounding Box for Enginerring Materials
In this project I have Using Deep Learning Techinques to predict bounding boxes for Engineering materials

# I. Problem Statement

In this project I have Using Deep Learning Techinques to predict bounding boxes for Engineering materials 

The dataset is available in link below:

https://drive.google.com/file/d/1IsK268zLnXB2Qq0X2LgNDwBZRuVwvjRx/view?usp=sharing

Jupyter Notebook link:



Major Highlights:

-  Used Panoptic DETR is used to generated ground truth of stuff for images givens
-  Final dataset is combined with coco stuff dataset
-  Fine tuning is done on the final dataset



- Only 5 epochs are used to achive the result




## Running the code

For Training, you can use download the code

git clone https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning.git 

Alternatively , if you directly want to use the Jupyter Notebook, please take API from the google drive link:

https://drive.google.com/drive/folders/1YTvb7V0eDfn5MZwBbc4msFkWKH5ArotI?usp=sharing 


## II. Data Cleaning and Data Loading

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
49 - 182 - Coco Stuff objects mapped using the formula (4892 + coco stuff category id - 43).


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


As our dataset may not have all the stuff, so the coco validation stuff dataset is added for training and for this also an intermediate
csv file is created

While generating csv files, Any of the image which is not RGB  (like RGBA or grey image) or images in WEBP format are ignored.

All the intermediate csv files are combined using into two jsons, custom-val.json and custom-train.json
using Jupyter notebook. While combining al lthe csv files , each image is given a unique id with format category name concated with image id.



## Training


As the combined image size is very large, it is split into multiple zip files and then unzipped using the structure below


/content/data/custom
├ annotations/  # JSON annotations
│  ├ annotations/custom_train.json
│  └ annotations/custom_val.json
├ train2017/    # training images
└ val2017/      # validation images



The DETR code used for training from repository 


Hyperparameters:

Number of queries: 30
Learning Rate: 1e-5




## Results:

The training is still going on as of today, I took some intermediate results after 6th epoch from validation Dataset as below



###i.

Ground truth with bounding boxes:

![image1_g](/part2/images/image_1_ground_truth.png)


Predicted bounding boxes:

![image1_p](/part2/images/image_1_predicted.png)


###ii.

Ground truth with bounding boxes:

![image2_g](/part2/images/image_2_ground_truth.png)


Predicted bounding boxes:

![image2_p](/part2/images/image_2_predicted.png)



###iii.

Ground truth with bounding boxes:

![image3_g](/part2/images/image_3_ground_truth.png)


Predicted bounding boxes:

![image3_p](/part2/images/image_3_predicted.png)



###iv.

Ground truth with bounding boxes:

![image4_g](/part2/images/image_4_ground_truth.png)


Predicted bounding boxes:

![image4_g](/part2/images/image_4_predicted.png)


###v.

Ground truth with bounding boxes:

![image4_g](/part2/images/image_5_ground_truth.png)


Predicted bounding boxes:

![image5_g](/part2/images/image_5_predicted.png)



###vi.

Ground truth with bounding boxes:

![image6_g](/part2/images/image_6_ground_truth.png)


Predicted bounding boxes:

![image6_g](/part2/images/image_6_predicted.png)


###vii.

Ground truth with bounding boxes:

![image7_g](/part2/images/image_6_ground_truth.png)


Predicted bounding boxes:

![image8_p](/part2/images/image_6_predicted.png)


## Plots:

The following are plots from 6th epoch

### i. Loss Plot

![image9](/part2/images/loss_plot.png)


### ii. mAP Plot

![image10](images/mAP_plot.png]


### iii. Loss CE Plot

![image11](/part2/images/loss_ce_plot.png)

### iv. Loss Bounding Box Plot

![image12](/part2/images/loss_box_plot.png)


### v. Loss GIOU Plot

![image12](/part2/images/loss_giou_plot.png)


### vi. Class Error Plot

![image12](/part2/images/class_error_plot.png)

### vii. Cardinality Error Unscaled plot

![image13](/part2/images/cardinality_error_unscaled.png)






# IX. Conclusion















