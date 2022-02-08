import csv
import os
import ast
import cv2
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
def cv_draw(img_root,bboxes):
    img_name=os.path.basename(img_root)
    image = cv2.imread(img_root)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for boxes in bboxes:
        bbox = boxes[1:]
        label = str(boxes[0])
        label_size1 = cv2.getTextSize(label, font, 1, 2)
        text_origin1 = np.array([bbox[0], bbox[1] - label_size1[0][1]])
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      color=(0, 0, 255), thickness=2)
        cv2.imwrite('./{}'.format(img_name), image)

def get_bbox_img(bbox,img_path):
    img=cv2.imread(img_path)#whc
    # img = img.transpose(1, 0, 2)
    # img=img[:,::-1,:]
    h,w=img.shape[:2]
    # h,w=1,1
    if bbox[0]<1:
        bbox[0]=int(bbox[0]*w)
        bbox[1]=int(bbox[1]*h)
        bbox[2]=int(bbox[2]*w)
        bbox[3]=int(bbox[3]*h)
        bbox_img=img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    else:
        bbox_img=img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    return bbox_img

def get_bbox_img_ht(bbox,img_path):
    img=cv2.imread(img_path)#whc

    h,w=img.shape[:2]
    if h<=w:
        if bbox[0]<1:
            bbox[0]=int(bbox[0]*w)
            bbox[1]=int(bbox[1]*h)
            bbox[2]=int(bbox[2]*w)
            bbox[3]=int(bbox[3]*h)
            bbox_img=img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        else:
            bbox_img=img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    else:
        if bbox[0]<1:
            bbox[0]=int(bbox[1]*w)
            bbox[1]=int(bbox[0]*h)
            bbox[2]=int(bbox[3]*w)
            bbox[3]=int(bbox[2]*h)
            bbox_img=img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        else:
            bbox_img=img[bbox[0]:bbox[2],bbox[1]:bbox[3],:]
    return bbox_img

def read_annotation_file(path):
    try:
        dataframe = pd.read_excel(path)
    except:
        dataframe = pd.read_csv(path)
    return dataframe

def str2dict(x):
    return ast.literal_eval(x)

def split(x):
    return ast.literal_eval(x)

def xywh2xyxy(bboxes):
    return [bboxes[0],bboxes[1],bboxes[0]+bboxes[2],bboxes[1]+bboxes[3]]

def trans2os2d(output_csv,label_root,img_root,required_columns):
    imgs=os.listdir(img_root)
    imgs.remove('货架标记')
    org_df=read_annotation_file(label_root)
    org_df.region_shape_attributes=org_df.region_shape_attributes.apply(str2dict)
    org_df.region_attributes=org_df.region_attributes.apply(str2dict)
    # print(org_df.region_shape_attributes)
    print(type(org_df.region_shape_attributes[0]))
    # print(ast.literal_eval(org_df.region_shape_attributes))
    new_df=pd.DataFrame(columns=required_columns)
    new_df.split=org_df.filename.apply(lambda x:'train')
    new_df.lx=org_df.region_shape_attributes.apply(lambda x:x['x'])
    new_df.rx=org_df.region_shape_attributes.apply(lambda x:x['x']+x['width'])
    new_df.ty=org_df.region_shape_attributes.apply(lambda x:x['y'])
    new_df.by=org_df.region_shape_attributes.apply(lambda x:x['y']+x['height'])
    new_df.difficult=0
    new_df.gtbboxid=range(len(org_df))
    # new_df.classid=org_df.region_attributes.apply(lambda x:x['sqbbbtwqps'])
    new_df.classid=org_df.region_attributes.apply(lambda x:x['skus'])
    new_df.imageid=org_df.filename.apply(lambda x:x[:-4])
    print(new_df)
    new_df.to_csv('./os2d/skus_new.csv')
    # print(imgs)

def trans2keypoint(output_csv,label_root,img_root,required_columns):
    imgs=os.listdir(img_root)
    # imgs.remove('货架标记')
    org_df=read_annotation_file(label_root)
    print(len(org_df))
    # print(org_df.region_shape_attributes)
    org_df=org_df.drop(org_df[org_df.region_shape_attributes=='{}'].index)
    print(len(org_df))
    # org_df.to_csv('./keypoint/hj3.csv',index=False)
    org_df.region_shape_attributes=org_df.region_shape_attributes.apply(str2dict)
    org_df.region_attributes=org_df.region_attributes.apply(str2dict)
    # print(org_df.region_shape_attributes)
    print(type(org_df.region_shape_attributes[0]))
    # print(ast.literal_eval(org_df.region_shape_attributes))
    new_df=pd.DataFrame(columns=required_columns)
    new_df.split=org_df.filename.apply(lambda x:'train')
    new_df.x=org_df.region_shape_attributes.apply(lambda x:max(x['cx'],0))
    new_df.y=org_df.region_shape_attributes.apply(lambda x:max(x['cy'],0))
    # new_df.classid=org_df.region_attributes.apply(lambda x:x['sqbbbtwqps'])
    new_df.classid=org_df.region_attributes.apply(lambda x:x['point'])
    new_df.imageid=org_df.filename.apply(lambda x:x)
    print(new_df)
    new_df.to_csv(output_csv,index=False)
    # print(imgs)

def split_train_test(csv_path,output_csv,rate=0.2):
    df=pd.read_csv(csv_path)
    img_names=list(df.imageid.drop_duplicates())
    train_imgs,test_imgs=train_test_split(img_names,test_size=rate,random_state=2)
    # print(train_imgs,test_imgs)
    for i in range(len(df)):
        if df.loc[i,'imageid'] in test_imgs:
            df.loc[i,'split']='test'
    df.to_csv(output_csv,index=False)
    return


def norm_xy(img_root,label_root,output_csv):
    org_df=read_annotation_file(label_root)
    for i in range(len(org_df)):
        print(os.path.join(img_root,org_df.loc[i,'imageid']))
        img=cv2.imread(os.path.join(img_root,org_df.loc[i,'imageid']))
        h,w=img.shape[:-1]
        print('H:',h,'W:',w)
        org_df.loc[i,'x']=org_df.loc[i,'x']/w

        org_df.loc[i,'y']=org_df.loc[i,'y']/h

    org_df.to_csv(output_csv,index=False)

def norm_xyxy(img_root,label_root,output_csv):
    org_df=read_annotation_file(label_root)
    for i in range(len(org_df)):
        print(os.path.join(img_root,org_df.loc[i,'imageid']+'.jpg'))
        img=cv2.imread(os.path.join(img_root,org_df.loc[i,'imageid']+'.jpg'))
        h,w=img.shape[:-1]
        print('H:',h,'W:',w)
        org_df.loc[i,'lx']=org_df.loc[i,'lx']/w
        org_df.loc[i,'rx']=org_df.loc[i,'rx']/w
        org_df.loc[i,'ty']=org_df.loc[i,'ty']/h
        org_df.loc[i,'by']=org_df.loc[i,'by']/h
    org_df.to_csv(output_csv,index=False)
        #  print(h)
        #  print(w)
def de_norm_xyxy(img_root,label_root,output_csv):
    org_df=read_annotation_file(label_root)
    for i in range(len(org_df)):
        print(os.path.join(img_root,org_df.loc[i,'imageid']+'.jpg'))
        img=cv2.imread(os.path.join(img_root,org_df.loc[i,'imageid']+'.jpg'))
        h,w=img.shape[:-1]
        print('H:',h,'W:',w)
        org_df.loc[i,'lx']=org_df.loc[i,'lx']*w
        org_df.loc[i,'rx']=org_df.loc[i,'rx']*w
        org_df.loc[i,'ty']=org_df.loc[i,'ty']*h
        org_df.loc[i,'by']=org_df.loc[i,'by']*h
    org_df.to_csv(output_csv,index=False)

    return 
if __name__ == '__main__':
    required_columns = {"classid", "imageid", "x", "y", 'split'}
    lab_root='./标注/points.csv'
    image_root='./标注/images/'
    # a=pd.read_excel(lab_root)
    '''获取转化后的csv文件'''
    output_csv1='./data/p1.csv'
    trans2keypoint(output_csv=output_csv1,label_root=lab_root,img_root=image_root,required_columns=required_columns)
    '''完成bbox的归一化）'''
    output_csv2='./data/p2_norm.csv'
    norm_xy(image_root,output_csv1,output_csv2)
    '''完成训练、验证集划分'''
    output_csv3='./data/p3_norm_splited.csv'
    split_train_test(output_csv2,output_csv3,rate=0.2)