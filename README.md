# prep_rcnn_train
Training an RCNN

Put the Annotations in gt_orig folder and the Images in pre_Images folder

First Clear all the previous files using

./clean.sh

To run the file use 

./run.sh

The Prepared Data will be stored in Images and Annotations


# To Prepare Data for detectron training datat in coco json

Convert the txt file data to individual XML files

python toxml.py

Convert the xml files into a Single JSON file in COCO format

python xml2json.py

The following file created will be instances.json, which can be used for detectron training
