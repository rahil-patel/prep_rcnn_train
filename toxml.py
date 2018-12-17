
import pandas as pd
import os
import cv2
import xml.etree.ElementTree as ET
import xml.dom.minidom

from dicttoxml import dicttoxml

import json
import re
from collections import OrderedDict


#src = 'train_set'
ann_src = "Annotations"
img_src = "Images"

txts = os.listdir(ann_src)
imgs = os.listdir(img_src)
txt_files = [x for x in txts if x.endswith('_gt.txt')]
img_files = [x for x in imgs if x.endswith('.jpg')]
print(img_files[1])


def convert_pascal(df,txt,file_path):
#     data = {}
    data = OrderedDict()
    data['folder'] = img_src
    image_name = txt.split('_gt.txt')[0]+'.jpg'
    data['filename'] = image_name
#     data['path'] = 
    image_path = os.path.join(img_src,image_name)
    print(image_path)
    img = cv2.imread(image_path)
    h,w,c = img.shape
    shape = OrderedDict()
    shape['height'] = h
    shape['width'] = w
    shape['depth'] = c
    data['size'] = shape
    data['segmented'] = 0
#     print(data)
    count = 0
    for index,row in df.iterrows():
            
            object_lists = []
            single_object = OrderedDict()
            single_object['name'] = row['upc']
            boxes = OrderedDict()
            boxes['xmin'] = row['x']
            boxes['ymin'] = row['y']
            boxes['xmax'] = int(row['x']) + int(row['w'])
            boxes['ymax'] = int(row['y']) + int(row['h'])
            #print(boxes)
            single_object['bndbox'] = boxes
            data['object'+str(count)] = single_object
            count = count+1
#     print('data',data)

    xml = dicttoxml(data,custom_root='annotation',attr_type=False)
#     xml_string = xml.decode('utf-8')
    xml_string = re.sub(r"object\d+", "object", xml)
#     print(xml_string)
    output = open('xml_files/'+txt[:-4]+'.xml', 'w')
    output.write(xml_string)
    output.close()

#     xml = str(xml)

#     print(xml)    
#     myxml = ET.fromstring(xml)



#txt_files = ['220613_gt.txt']
for txt in txt_files:
    print(txt)
    file_path = os.path.join(ann_src,txt)
    try:
        df = pd.read_csv(file_path,sep=' ',header=None,dtype=str)
        df.columns = ['x','y','w','h','upc']
    
        convert_pascal(df,txt,file_path)
    except pd.io.common.EmptyDataError:
        continue

