import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import random
import shutil
import sys
import argparse
from tqdm import tqdm


gt_src = 'gt_orig'
gt_dst = 'Annotations'
img_src = 'pre_Images'
ground_dst = 'gt_images'
split_ann = 'Annotations'
split_img = "Images"
upc_brands = {}


brand = pd.read_csv("upc_brand.txt",skiprows=0,sep=" ",dtype=str)
brand.columns = ['upc','brand']
brand.head()
upc = brand['upc']
br = brand['brand']
for i in range(len(upc)):
    upc_brands[upc[i]] = br[i]


txt_files = os.listdir(gt_src)
#txt_files = ['836254_2017_10_30_2228_102270_201744_1_2487_6949740_20171030173656107_gt.txt']

def do_aug(img,img_name,dets_bb,split_num):
    image_shape = img.shape
    keypoints = []
    keypoints_on_images = []
    for i in range(dets_bb.shape[0]):
        x = dets_bb[i][0]
        y = dets_bb[i][1]
        keypoints.append(ia.Keypoint(x=x, y=y))

        x = dets_bb[i][2]
        y = dets_bb[i][3]
        keypoints.append(ia.Keypoint(x=x, y=y))

    keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=img.shape))

#     seq_det = iaa.Sequential([iaa.Affine(scale=(0.5,1.5)),iaa.Add((-40, 40))])
    seq_det = iaa.Sequential([iaa.Add((-40, 40))])
    
    images_aug = seq_det.augment_image(img)
    keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)
    im_name = img_name+'_sp_'+str(split_num)+'.jpg'
    image_txt = img_name+'_sp_'+str(split_num)+'_gt.txt'
    #print("Image Name::",image_txt)
    figname = os.path.join(split_img,im_name)
    txt_name = os.path.join(split_ann,image_txt)
    #print("text_name::",txt_name)
    cv2.imwrite(figname,images_aug)
    
    #keypoints = keypoints_aug[0].keypoints
    #print(keypoints)
    
    file_arr=np.loadtxt(txt_name, dtype='str', ndmin=2 )
    #print(txt_name)#,file_arr,file_arr.shape[0])
    file_new_arr=[]
    for box_num in range(file_arr.shape[0]):
        row = list(file_arr[box_num])
#         print(keypoints[2*box_num].x,keypoints[2*box_num].y,keypoints[2*box_num + 1].x,keypoints[2*box_num + 1].y,image_shape[1],image_shape[0])
        #print(row)
#         if((keypoints[2*box_num].x > -10) and (keypoints[2*box_num].y > -10) and (keypoints[2*box_num + 1].x < image_shape[1] + 10) and (keypoints[2*box_num + 1].y < image_shape[0] + 10) ):
#             row[0]= keypoints[2*box_num].x
#             row[1]= keypoints[2*box_num].y
#             row[2]= keypoints[2*box_num + 1].x - keypoints[2*box_num].x
#             row[3]= keypoints[2*box_num + 1].y - keypoints[2*box_num].y
#             file_new_arr.append(row)
    file_new_arr = np.array(file_arr, ndmin=2)
    if(file_new_arr.shape[0]>0 and file_new_arr.shape[1]>0):
        #print(file_new_arr)
        np.savetxt(txt_name,file_new_arr,delimiter=" ", fmt="%s")

def get_bbox(df):
    objects = []
    for index,row in df.iterrows():
        obj_struct = {}
        obj_struct['bbox'] = [int(row['x']),int(row['y']),int(int(row['x'])+int(row['w'])),int(int(row['y'])+int(row['h']))]
        #print(row['brand'])      
        obj_struct['name'] = [row['brand']]  
        objects.append(obj_struct)
    
    R = [obj_inside for obj_inside in objects ]
    dets_bb = np.array([x['bbox'] for x in R])
    dets_class = np.array([x['name'] for x in R])
    
    return R,dets_bb,dets_class

def split_augmentation(im,dets_bb,dets_class,split_size,overlap_ratio,image_id,d_aug):
    #print(image_id)
    size_h,size_w, channels= im.shape
    split_size_h = size_h/split_size[0]
    split_size_w = size_w/split_size[1]

    split_num=0
    im_split = [[] for _ in range(split_size[0] * split_size[1])]


    for split_row in range(int(split_size[0])):
        for split_col in range(int(split_size[1])):
            split_dets_count = 0
            split_dets = []
            split_cls = []
            quad_row_start = int(max(0,split_row*(size_h/split_size[0]) - int(overlap_ratio*(size_h/split_size[0]))))
            quad_row_end = int(min(size_h,(split_row+1)*(size_h/split_size[0]) + int(overlap_ratio*(size_h/split_size[0]))))
            quad_col_start = int(max(0,split_col*(size_w/split_size[1]) - int(overlap_ratio*(size_w/split_size[1]))))
            quad_col_end = int(min(size_w,(split_col+1)*(size_w/split_size[1]) + int(overlap_ratio*(size_w/split_size[1]))))

            im_split[split_num] = im[ quad_row_start:quad_row_end , quad_col_start:quad_col_end , : ]

            for box,cls_name in zip(dets_bb,dets_class):
                if(box[0]>quad_col_start and box[1]>quad_row_start and box[2]<quad_col_end and box[3]<quad_row_end):
                    newbox=np.copy(box)
                    newbox[0] = newbox[0] - quad_col_start
                    newbox[2] = newbox[2] - quad_col_start
                    newbox[1] = newbox[1] - quad_row_start
                    newbox[3] = newbox[3] - quad_row_start

                    newbox[2] = newbox[2] - newbox[0]
                    newbox[3] = newbox[3] - newbox[1]

                    if split_dets_count==0:
                        split_dets=newbox
                        split_cls=np.asarray([cls_name])
                    else:
                        split_dets = np.vstack((split_dets,newbox)) 
                        split_cls = np.vstack((split_cls,[cls_name]))  
                    split_dets_count = split_dets_count + 1

            if split_dets_count ==1:
                split_dets = [split_dets]
            if split_dets_count>0:
                split_dets = np.asarray(split_dets,dtype=str)
                split_cls = np.asarray(split_cls,dtype=str)
                bbox_quad=np.hstack((split_dets,split_cls))
                bbox_quad=np.array(bbox_quad,ndmin=2)                   
                newfilename=os.path.join(split_ann,image_id+'_sp_'+str(split_num)+'_gt.txt')
                np.savetxt(newfilename,bbox_quad,delimiter=" ", fmt="%s")
                figname=os.path.join(split_img,image_id+'_sp_'+str(split_num)+'.jpg')
                cv2.imwrite(figname,im_split[split_num])
                if d_aug==1:
                    do_aug(im_split[split_num],image_id,dets_bb,split_num)
            split_num = split_num + 1


def file_aug(df,txt_file,img,img_name,split_size,d_aug):
    size_h,size_w,channels = img.shape
    area_im = size_h * size_w
    aug_threshold = 0.003
    overlap_ratio = 0.2
    R,dets_bb,dets_class = get_bbox(df)
    bool_split_aug=False
    
    for box in dets_bb:
        area_box = (box[2]-box[0]) * (box[3]-box[1])
        if ((float(area_box)/float(area_im))<aug_threshold):
            bool_split_aug=True
            break
    
    if bool_split_aug:
        #split_true += 1  
        filename = txt_file.split("_gt.txt")[0]
        split_augmentation(img,dets_bb,dets_class,split_size,overlap_ratio,filename,d_aug)


def groundtruth(df,txt_file,img,img_name):
    if not df.empty:
        for index,row in df.iterrows():
            x2 = int(row['x'])+int(row['w'])
            y2 = int(row['y'])+int(row['h'])
            cv2.rectangle(img,(int(row['x']),int(row['y'])),(x2,y2),(0,255,0),4)
            cv2.putText(img,row['brand'],(int(row['x']),int(row['y'])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        img_dst = os.path.join(ground_dst,img_name)
        cv2.imwrite(img_dst,img)

def counting(df):
    uniq_brand = br.unique()
    #print(uniq_brand)
    #print(df.columns)
    count = [0]*len(uniq_brand)
    for i in range(len(uniq_brand)):
        count[i] = len(df.loc[df['brand']==uniq_brand[i]])
    for i in range(len(uniq_brand)):
        print(str(uniq_brand[i])+":"+str(count[i]))
    print("Total Number of Samples::",sum(count))


def split_count_process(split_option):
    train_list = []
    test_list = []
    count_df = pd.DataFrame()
    file_count = 0
    new_files = os.listdir(split_ann)
    #print(new_files)
    imageset = [ x.split("_gt.txt")[0] for x in new_files]
    with open('train.txt', mode='wt') as myfile:
        myfile.write('\n'.join(imageset))


    if split_option ==1:
	for txt_file in new_files:
        	files = os.path.join(split_ann,txt_file)
        	try:
            		df = pd.read_csv(files,sep=' ',dtype=str,header=None)
            		df.columns = ['x','y','w','h','brand']
            		count_df = count_df.append(df,ignore_index=True)

        	except pd.io.common.EmptyDataError:
            		e=0
    

    	print("Results After Augmentations::")
    	counting(count_df)

def preprocess(df,img_shape):
    size_h,size_w,channels = img_shape
    for index,row in df.iterrows():
        if int(row['x'])<=0:
            row[0]=str(3)
        
        if int(row['y'])<=0:
            row[1]=str(3)
        
        if int(row['x'])+int(row['w'])>=size_w:         
            row['w']=str(int(row['w']) - abs((size_w - (int(row['x'])+int(row['w'])))) - 3) 
        
        if int(row['y'])+int(row['h'])>=size_h:
            row['h']=str(int(row['h']) - abs((size_h - (int(row['y'])+int(row['h'])))) - 3)
    return df

count_df = pd.DataFrame()
def integration(split_s,gt_option,d_aug,split_option):
    count_df = pd.DataFrame()
    file_count = 0
    for txt_file in tqdm(txt_files):
        path = os.path.join(gt_src,txt_file)
        dst = os.path.join(gt_dst,txt_file)
        #print(path)
        try:
            #print(path)
	    df = pd.read_csv(path,sep=' ',dtype=str,header=None)
            df.columns = ['x','y','w','h','upc']
            
            df = pd.merge(df,brand,how='left',on='upc')
            df.dropna(inplace=True)
            #print(df)
            if df.empty:
                df.to_csv(dst,sep=" ",index=False,header=False,columns = ['x','y','w','h','brand'])
                file_count += 1
            elif not df.empty:
                img_name = txt_file.split("_gt.txt")[0]+'.jpg'
                #print(img_name)
                img_path = os.path.join(img_src,img_name)
                img = cv2.imread(img_path)
                shutil.copy(img_path,os.path.join(split_img,img_name))
                df = preprocess(df,img.shape)
                df.to_csv(dst,sep=" ",index=False,header=False,columns = ['x','y','w','h','brand'])
                file_count += 1
                
                gt_image = img.copy()
                if gt_option == 1 :
                    groundtruth(df,txt_file,gt_image,img_name)
		if split_option == 1:
		    file_aug(df,txt_file,img,img_name,split_s,d_aug)
                count_df = count_df.append(df,ignore_index=True)
        except pd.io.common.EmptyDataError:
            e=0
            #print(path, " is empty and has been skipped.")
    counting(count_df)
    print("Total Number of Files::",file_count)
    imageset = [x.split("_gt.txt")[0] for x in txt_files]
    #if split_option ==1:
    split_count_process(split_option)


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Train Directory')
    parser.add_argument('--gt',dest='gt_option',type=int,default=1,help='Perform groundtruths or Not')
    parser.add_argument('--split_size',dest='split_s',default=[4,4],nargs='+',help='Number of splitting windows',type=int)
    parser.add_argument('--aug',dest='d_aug',help='perform aug or not',type=int,default=0)
    parser.add_argument('--split',dest='split_option',help='Perform split or not',type=int,default=0)
    args = parser.parse_args()
    print(args.d_aug)
    integration(args.split_s,args.gt_option,args.d_aug,args.split_option)
