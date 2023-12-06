import pandas as pd

data_path = '/plant-pathology-2020-fgvc7/'
img_dir = '/plant-pathology-2020-fgvc7/images/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')