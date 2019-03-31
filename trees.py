#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind_from_stats
import random


# In[2]:


#sys.argv[1] = 'trainingSet.csv'
#sys.argv[2] = 'testSet.csv'

trainingDataFilename = sys.argv[1]
testDataFilename = sys.argv[2]
modelIdx = int(sys.argv[3])


# In[3]:


DEPTH_LIMIT = 8
SAMPLE_LIMIT = 50
class TreeNode(object):
    def __init__(self, class_label, lineage):
        self.val = class_label
        self.lineage = lineage
        self.left = None
        self.right = None

    def is_leaf(self):
        return ((self.left is None) or (self.right is None))

    def expand(self, data, features, modelIdx, depth=DEPTH_LIMIT):
        class_label = int(len(data[data['decision'] == 1]) > len(data[data['decision'] == 0]))
        if modelIdx == 1 or modelIdx == 2:
            available_features = [f for f in features if f not in self.lineage]
        else:
            rf_sample = [i for i in range(0, int(len(features)))]
            np.random.shuffle(rf_sample)
            rf_sample = rf_sample[0:int(np.sqrt(len(features)))]
            rf_features = []
            for i in rf_sample:
                rf_features.append(features[i])
            #print rf_features
            available_features = [f for f in rf_features if f not in self.lineage]
        if len(self.lineage) < depth - 1 and len(data) > 50:
            ginis = {k: get_gini_gain(data, k) for k in available_features}
            best_feature = max(ginis, key=ginis.get)
            left_data = data[data[best_feature] == 0]
            right_data = data[data[best_feature] == 1]
            
            self.val = best_feature
            self.left = TreeNode(class_label, self.lineage + [best_feature])
            self.left.expand(left_data, features, modelIdx, depth=depth)
            self.right = TreeNode(class_label, self.lineage + [best_feature])
            self.right.expand(right_data, features, modelIdx, depth=depth)
        else:
            self.val = class_label
            
def gini(data):
    total = len(data)
    pos = 1.0 * len(data[data['decision'] == 1])/(1.0 * total) if len(data[data['decision'] == 1]) > 0 else 0.0
    neg = 1.0 * len(data[data['decision'] == 0])/(1.0 * total) if len(data[data['decision'] == 0]) > 0 else 0.0
    return 1.0 - pos * pos - neg * neg

def get_gini_gain(data, attr):
    total = len(data)
    pos = 1.0 * len(data[data[attr] == 1])/(1.0 * total) if len(data[data[attr] == 1]) > 0 else 0.0
    neg = 1.0 * len(data[data[attr] == 0])/(1.0 * total) if len(data[data[attr] == 0]) > 0 else 0.0
    gain = gini(data) - pos * gini(data[data[attr] == 1]) - neg * gini(data[data[attr] == 0])
    return gain


# In[4]:


def decisionTree(trainingSet, testSet, keys):
    root = TreeNode(-1, [])
    root.expand(trainingSet, keys, 1, DEPTH_LIMIT)
    return root

def predict(node, data):
    if node.is_leaf():
        return node.val
    else:
        if data[node.val] == 0:
            return predict(node.left, data)
        else:
            return predict(node.right, data)

def get_accuracy_dt(root, trainingSet, testSet):
    count_train, total_train = 0, len(trainingSet)
    count_test, total_test = 0, len(testSet)
    
    train_labels = trainingSet['decision']
    test_labels = testSet['decision']
    
    trainingSet = trainingSet.drop('decision', axis=1)
    testSet = testSet.drop('decision', axis=1)
    
    predictions = []
    Y = np.array(train_labels)
    # Training accuracy
    for index, row in trainingSet.iterrows():
        predictions.append(predict(root, row)) 
        if int(predictions[index]) == Y[index]:
            count_train += 1
    
    training_accuracy = 1.0 * count_train/total_train
    print 'Training Accuracy DT:', '%.2f' % training_accuracy
    
    predictions = []
    Y = np.array(test_labels)

    # Test accuracy
    for index, row in testSet.iterrows():
        predictions.append(predict(root, row))
        if predictions[index] == Y[index]:
            count_test += 1

    test_accuracy = 1.0 * count_test/total_test
    print 'Test Accuracy DT:', '%.2f' % test_accuracy


# In[5]:


def bagging(trainingSet, testSet, keys):
    root_bagging = []
    for i in range(30):
        train = trainingSet.sample(frac = 1.0, replace=True)
        root = TreeNode(-1, [])
        root.expand(train, keys, 2, DEPTH_LIMIT)
        root_bagging.append(root)
    return root_bagging

def get_accuracy_bagging(root, trainingSet, testSet):
    count_train, total_train = 0, len(trainingSet)
    count_test, total_test = 0, len(testSet)
    
    train_labels = trainingSet['decision']
    test_labels = testSet['decision']
    
    trainingSet = trainingSet.drop('decision', axis=1)
    testSet = testSet.drop('decision', axis=1)
    
    predictions = [0 for i in range(len(trainingSet))]
    
    Y = np.array(train_labels)
    # Training accuracy
    for r in root:
        for index, row in trainingSet.iterrows():
            predictions[index] += int(predict(r, row))
    #print predictions

    for i in range(len(trainingSet)):
        if predictions[i] > 15 and Y[i] == 1:
            count_train += 1
        elif predictions[i] <= 15 and Y[i] == 0:
            count_train += 1
    
    training_accuracy = 1.0 * count_train/total_train
    print 'Training Accuracy BT:', '%.2f' % training_accuracy
    
    predictions = [0 for i in range(len(testSet))]
    Y = np.array(test_labels)

    # Test accuracy
    for r in root:
        for index, row in testSet.iterrows():
            predictions[index] += int(predict(r, row))

    for i in range(len(testSet)):
        if predictions[i] > 15 and Y[i] == 1:
            count_test += 1
        elif predictions[i] <= 15 and Y[i] == 0:
            count_test += 1

    test_accuracy = 1.0 * count_test/total_test
    print 'Test Accuracy BT:', '%.2f' % test_accuracy


# In[6]:


def randomForests(trainingSet, testSet, keys):
    root_rf = []
    for i in range(30):
        train = trainingSet.sample(frac = 1.0, replace=True)
        root = TreeNode(-1, [])
        root.expand(train, keys, 3, DEPTH_LIMIT)
        root_rf.append(root)
        #get_accuracy_dt(root, trainingSet, testSet)
    return root_rf
    
def get_accuracy_rf(root, trainingSet, testSet):
    count_train, total_train = 0, len(trainingSet)
    count_test, total_test = 0, len(testSet)
    
    train_labels = trainingSet['decision']
    test_labels = testSet['decision']
    
    trainingSet = trainingSet.drop('decision', axis=1)
    testSet = testSet.drop('decision', axis=1)
    
    predictions = [0 for i in range(len(trainingSet))]
    
    Y = np.array(train_labels)
    # Training accuracy
    for r in root:
        for index, row in trainingSet.iterrows():
            predictions[index] += int(predict(r, row))
    #print predictions

    for i in range(len(trainingSet)):
        if predictions[i] > 15 and Y[i] == 1:
            count_train += 1
        elif predictions[i] <= 15 and Y[i] == 0:
            count_train += 1
    
    training_accuracy = 1.0 * count_train/total_train
    print 'Training Accuracy RF:', '%.2f' % training_accuracy
    
    predictions = [0 for i in range(len(testSet))]
    Y = np.array(test_labels)

    # Test accuracy
    for r in root:
        for index, row in testSet.iterrows():
            predictions[index] += int(predict(r, row))

    for i in range(len(testSet)):
        if predictions[i] > 15 and Y[i] == 1:
            count_test += 1
        elif predictions[i] <= 15 and Y[i] == 0:
            count_test += 1

    test_accuracy = 1.0 * count_test/total_test
    print 'Test Accuracy RF:', '%.2f' % test_accuracy


# In[7]:


trainingSet = pd.read_csv(trainingDataFilename)
testSet = pd.read_csv(testDataFilename)

keys = trainingSet.keys()
keys = keys.drop('decision')

if modelIdx == 1:
    root = decisionTree(trainingSet, testSet, keys)
    get_accuracy_dt(root, trainingSet, testSet)
elif modelIdx == 2:
    root = bagging(trainingSet, testSet, keys)
    get_accuracy_bagging(root, trainingSet, testSet)
elif modelIdx == 3:
    root = randomForests(trainingSet, testSet, keys)
    get_accuracy_rf(root, trainingSet, testSet)
else:
    print 'modelIdx error'


# In[ ]:




