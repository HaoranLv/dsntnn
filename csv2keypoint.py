import csv
import os
import pandas as pd
import shutil
import numpy as np
import ast
from PIL import Image
points=['point1','point2','point3','point4','point5','point6','point7','point8']
# classes=['80120175003', '80140375001', '80440170001', '80101075001', '80400136001', '80400171501', '80400936001', '80440152001', '80400153001', '80400371501', '80720232001', '80440110101', '80101010101', '80400110701']
# classes=['AHyzhywqps', 'ksbtwsdqps', 'nfsqrxxjwsdqps', 'qqdqtwsdqps', 'sqbbbtwqps', 'tqdbtwsdqps', 'sqbbxgwqps', 'nfsqfxbtwsdqps', 'yqslkmjwsdqpsA', 'yqslbtwsdqpsA', 'sodazyqtqps', 'wxqpbtwqps', 'yqslbtwsdqps', 'xcgsbtwwtqps', 'xcjfptwwtqps', 'wxqpbxgwqps', 'HPhppnmfwqps', 'nfsqcjygwsdqps', 'ksnmwsdqps', 'tyAHnmwfjyl', 'xchyyzwwtqps', 'yqslkmjwsdqps', 'kkkltwyl', 'yqslrsjwsdqps', 'yqslsmzsdqps', 'ksllwsdqps', 'xchylzwwtqps', 'xcblbxgrsjwwtqps', 'yqdxywsdqps', 'AHbtwlcwqps', 'qtssmtwyl', 'lqdlzwsdqps', 'jqdjjwsdqps', 'wxqpmywqps', 'qpsbptwyl', 'tyAHpgcfjyl', 'qnsnmwyl', 'nfsqmjtwsdqps', 'yqslxhptwsdqps', 'yrnrwyl', 'HPhppsmtfwqps', 'sodathytqps', 'sodakmjxrzqps', 'yrsmtwyl', 'ssnmwsdqps']
def mov(file_name, base_dir='/Users/lvhaoran/Downloads/summitimg', target='./coco_train2/images'):
    shutil.move(os.path.join(base_dir, file_name), os.path.join(target, file_name))
    return

def get_classes(csv_path):
    res=[]
    df=pd.read_csv(csv_path,header=None)
    for i in range(len(df)):
        # print(df.loc[i,'region_attributes'])
        if df.iloc[i,0] not in res:
            res.append(df.iloc[i,0])
        else:
            print(df.iloc[i,0])
    return res

def trans(csv_root, txt_root,base_image_root,target_image_root,task='train'):
    # csv_reader=csv.reader(open(csv_root,'r'))
    df=pd.read_csv(csv_root)
    # df=df[df.gtbboxid<=14090]
    res_dic={}
    for i in range(len(df)):
        if df.loc[i]['split']==task:
            if df.loc[i]['imageid'] not in res_dic.keys():
                res_dic[df.loc[i]['imageid']]={}
            x=df.loc[i]['x']
            y=df.loc[i]['y']
            res_dic[df.loc[i]['imageid']][df.loc[i]['classid']]=[x,y]
    for k in res_dic.keys():
        if len(res_dic[k].keys())==8:
            lab=[]
            for i in range(8):
                lab.append(res_dic[k][points[i]])
            lab=np.array(lab)
            np.save(txt_root+'/'+k[:-4]+'.npy', lab)
            try:
                print(k)
                if task=='train':
                    mov(k,base_image_root,target=target_image_root)
                else:
                    mov(k,base_image_root,target=target_image_root)
            except:
                # print('false')
                continue

def gen_classes(csv_root,output_csv="class_mappings.csv"):
    classes=[]
    '''网页标注工具生成的csv使用此函数转化'''
    csv_class = open(output_csv, "a")
    vggcsv=csv.reader((open(csv_root,'r')))
    # vggcsv=pd.read_csv(csv_root)
    # vggcsv=vggcsv.drop(vggcsv[vggcsv.region_shape_attributes=='{}'].index)
    
    for i,v in enumerate(vggcsv):
        if i==0:
            continue
        v[6]=ast.literal_eval(v[6])
        class_name=v[6]['name']
        print(class_name)
        if class_name not in classes:
            classes.append(class_name)
            line_class = str(class_name) + ',' + str(len(classes)) + "\n"
            csv_class.write(line_class)
    return

if __name__ == '__main__':
    # gen_classes('./haitian/hj3.csv')
    # classes=get_classes('./class_mappings.csv')
    # # print(classes)
    # for i in range(len(classes)):
    #     classes[i] = int(classes[i])
    # print(classes)

    csv_root='./data/p3_norm_splited.csv'
    base_image_root='./data/images'
    train_lab_root='./data/tv/labels/train'
    train_image_root='./data/tv/images/train'
    test_lab_root='./data/tv/labels/val'
    test_image_root='./data/tv/images/val'
    roots=[train_image_root,test_image_root,train_lab_root,test_lab_root]
    for i in roots:
        if not os.path.exists(i):
            os.makedirs(i)
    trans(csv_root,txt_root=train_lab_root,base_image_root=base_image_root,target_image_root=train_image_root,task='train')
    trans(csv_root,txt_root=test_lab_root,base_image_root=base_image_root,target_image_root=test_image_root,task='test')