#!/usr/bin/python
#-*- coding:UTF-8 -*-

##################################################### 苏州 SAIL LAB ######################################################################
################################################## 机器学习研发工程师 ####################################################################
##################################### 利用灰度共生矩阵提取特征, 以供后续分类识别任务 #####################################################

import os
import pandas as pd
from skimage import io
from skimage.color import rgb2grey
from skimage.feature import greycomatrix, greycoprops

images_path = 'V:\\......\\......\\dataset'               # 存放待提取特征的图像数据集路径
dataset_csv = 'V:\\......\\......\\predicted.csv'         # Ground Truth保存于CSV的路径

class dataset:
    def __init__(self, img_pth, csv_pth):
        self.img_pth = img_pth
        self.csv_pth = csv_pth

    # 读取数据函数
    def read_data(self):
        ground_truth = {}
        df = pd.read_csv(self.csv_pth)
        image_name = [i for i in df['image_name']]                  # 利用pandas将csv中image name抽取出来
        new_df = df[['image_name', 'f_1']]                          # 将image name对应的标注信息抽取出来
        for i in range(0, len(new_df)):
            img_name = new_df.loc[i][0]
            label = new_df.loc[i][1]
            ground_truth[img_name] = label
        
        return image_name, ground_truth                       

    # 利用灰度共生矩阵, 对图像进行纹理特征提取
    def calculation(self, image_list):                              
        scores = {}                                                 # 用于存放 灰度共生矩阵(GCLM) 的'dissimilarity','homogeneity'.. 等 计算值 
        for img_name in image_list:
            image_pth = os.path.join(self.img_pth, img_name)
            image = io.imread(image_pth)
            image_s = image[:,:, 0]
            glcm = greycomatrix(image_s, distances=[1], angles=[0], levels=None, symmetric=True, normed=True)
            scores[img_name] = [float(greycoprops(glcm, x)) for x in ['dissimilarity','homogeneity', 'energy', 'correlation', 'ASM']]
        
        return scores

    # 将计算统计结果, 作为特征, 保存写入txt中
    def saved_dataset(self, dict_feature, dict_labels):                    
        for content in dict_feature:                               # dict_feature 是一个字典dict, key是image name, value是一个列表list存放GCLM计算统计的值
            for img_name in dict_labels:                           # dict_labels 是image name对应的Ground truth(GT)
                if content==img_name:
                    dict_feature[content].append(dict_labels[img_name])   # 添加GT信息至, dict_feature中value里

        # 将包含GCLM计算统计值、GT信息的dict_feature内容, 保存写入txt中
        saved_pth = os.path.join('V:\\Coding\\machine_learning', 'predict_dataset.txt')
        txt_file = open(saved_pth, 'w', encoding='UTF-8')
        for i in dict_feature:
            for index in range(0, len(dict_feature[i])):
                if index==0:
                    info = i
                else:
                    info += ' ' + str(dict_feature[i][index])
            info += '\n'
            txt_file.writelines(info)

def main():

    # 1st--首先调用上面类中函数, 将待识别的数据、信息找出来.
    work = dataset(images_path, dataset_csv)
    imgs, labels = work.read_data()

    # 2nd--将待提取数据送入算法中, 进行计算.
    scores = work.calculation(imgs)

    # 3rd--将计算求取的值, 作为特征保存写入一个txt文件中.
    work.saved_dataset(scores, labels)

if __name__ == '__main__':
    main()