import os
from sklearn.preprocessing import MinMaxScaler
from unittest import result

data_file = ''
num_classes = 2
class_names = ['closing_price', 'volume']
train_test_ratio = .10

EPOCHS = 20
BATCH_SIZE = 32

# optimizer = "sgd"
optimizer = "adam"
# optimizer = "rmsprop"

learning_rate = 0.01
model_name = "BiLSTM"

# paths and directories
result_dir = '/content/drive/MyDrive/BS-23/datasets/deep-learning-model-evaluation/results'
train_dir = "/content/train"
valid_dir = "/content/valid"
test_dir = "/content/test"

# train_dir = "/content/Dog_vs_Cat/train"
# valid_dir = "/content/Dog_vs_Cat/valid"
# test_dir = "/content/Dog_vs_Cat/test"
scaler = MinMaxScaler(feature_range=(0,1))

version = '1.0'
