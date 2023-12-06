from torch.utils.data import Dataset
import numpy as np
import cv2

class ImageDataset(Dataset):
    # 초기화 메서드(생성자)
    def __init__(self, df, img_dir='./', transform=None, is_test=False):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
    
    # 데이터셋 크기 반환 메서드
    def __len__(self):
        return len(self.df)
    
    # 인덱스(idx)에 해당하는 데이터 변환 메서드
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]                  # 이미지 ID
        img_path = self.img_dir + img_id + '.jpg'      # 이미지 파일 경로
        image = cv2.imread(img_path)                   # 이미지 파일 읽기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상 보정
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        if self.is_test:
            return image
        else:
            label = np.argmax(self.df.iloc[idx, 1:5])
            return image, label