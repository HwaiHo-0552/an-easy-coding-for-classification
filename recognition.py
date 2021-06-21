#!/usr/bin/python
#-*- coding:UTF-8 -*-

####################################################### 苏州 SAIL LAB ##################################################################
###################################################### 机器学习研发工程师 ##############################################################
########################################################## 分类识别  ###################################################################

from sklearn import svm
from sklearn.metrics import accuracy_score

dataset_path = 'V:\\......\\......\\......txt'                                  # 待training的数据, 已通过code写入保存在txt中
predicted = 'V:\\......\\......\\......txt'                                     # 待predicted的数据, 已通过code写入保存在txt中

class read_dataset:
    def __init__(self, data_pth):
        self.data_pth = data_pth

    def reading(self):
        features = {}
        labels = {}
        with open(self.data_pth, 'r+', encoding='UTF-8') as f:                  # 打开txt文件, 一行行的读取
            lines = f.readlines()

        for content in lines:
            content = content.strip()
            info = content.split(' ')
            img_name = info[0]
            features[img_name] = info[1:-1]
            labels[img_name] = info[-1]

        return features, labels                                                # 返回特征和GT

class recognition:
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def training(self):
        inputs = [self.dataset[i] for i in self.dataset]
        gt = [self.label[i] for i in self.label]

        classifier = svm.LinearSVC(C=1.0, max_iter=10000)                     # 分类器为SVM
        classifier.fit(inputs, gt)

        return classifier                                                     # 返回训练完成后的分类器

def main():
    # 1st--读取数据
    work = read_dataset(dataset_path)
    features, labels = work.reading()

    # 2nd--将特征和GT送入分类中, 进行训练
    recog = recognition(features, labels)
    classifier = recog.training()

    # 3rd--已完成了分类器的训练, 读入待预测的数据
    work_p = read_dataset(predicted)
    features_p, labels_p = work_p.reading()
    inputs = [features_p[i] for i in features_p]
    gt = [labels_p[i] for i in labels_p]
    
    # 4th--预测
    preds = classifier.predict(inputs)

    # 5th--统计结果
    scores = accuracy_score(preds, gt)
    print(scores)

if __name__ == '__main__':
    main()