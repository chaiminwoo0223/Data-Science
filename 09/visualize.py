from data import data_path, train
from utils import show_image, show_pie

labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
length = []

# 이미지
for label in labels:
    label_data = train.loc[train[label]==1]
    length.append(len(label))
    last_img_ids = label_data['image_id'][-12:]
    last_img_paths = [f'{data_path}images/{img_id}.jpg' for img_id in last_img_ids]
    show_image(last_img_paths, save_name= label + '.jpg')

# 파이
show_pie(length, labels)