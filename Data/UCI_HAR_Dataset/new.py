import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
# 定义数据类:训练数据+测试数据+特征
class activity_data:
    features = None
    train_data = None
    train_label = None
    test_data = None
    test_label = None
    label_name = None
    root = 'Data/UCI_HAR_Dataset/'
    # 初始化
    def __init__(self):
        self.ReadData()
    # 读取数据文件
    def ReadData(self):
        features = pd.read_csv(self.root + 'features.txt', sep='\s+', index_col=0, header=None)
        self.train_data = pd.read_csv(self.root + 'train/X_train.txt', sep='\s+', names=list(features.values.ravel()))
        self.test_data = pd.read_csv(self.root + 'test/X_test.txt', sep='\s+', names=list(features.values.ravel()))
        self.train_label = pd.read_csv(self.root + 'train/Y_train.txt', sep='\s+', header=None)
        self.test_label = pd.read_csv(self.root + 'test/Y_test.txt', sep='\s+', header=None)
        self.label_name = pd.read_csv(self.root + 'activity_labels.txt', sep=' ', header=None)
    #

data = activity_data()

data.train_data.head()

data.train_label.head()

data.test_data.head()

data.test_label.head()

data.train_label[0].value_counts()

data.test_label[0].value_counts()

rfc = RandomForestClassifier()

rfc = rfc.fit(data.train_data,data.train_label)

result = rfc.score(data.test_data,data.test_label)

result

rfc.estimators_

rfc.classes_

rfc.n_classes_

prediction = rfc.predict(data.test_data)

prediction

prediction_new = rfc.predict_proba(data.test_data)[:,:]

prediction_new

print('各标签的重要性')

rfc.feature_importances_