from data import img_dir, train, test, submission
from sklearn.model_selection import train_test_split
from transforms import transform_train, transform_test
from dataset import ImageDataset
from torch.utils.data import DataLoader
from seed_worker import seed_worker
import torch

# 하이퍼파라미터
g = torch.Generator()
g.manual_seed(0)
batch_size = 32

# 훈련 데이터, 검증 데이터 분리
train, valid = train_test_split(train,
                                test_size=0.1,
                                stratify=train[['healthy', 'multiple_diseases', 'rust', 'scab']],
                                random_state=10)

# 데이터셋 생성
dataset_train = ImageDataset(train, img_dir=img_dir, transform=transform_train)
dataset_valid = ImageDataset(valid, img_dir=img_dir, transform=transform_test)
dataset_test = ImageDataset(test, img_dir=img_dir, transform=transform_test, is_test=True)
dataset_tta = ImageDataset(test, img_dir=img_dir, transform=transform_train, is_test=True)

# 데이터로더 생성
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                          worker_init_fn=seed_worker, generator=g)
loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, 
                          worker_init_fn=seed_worker, generator=g)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, 
                         worker_init_fn=seed_worker, generator=g, num_workers=2)
loader_tta = DataLoader(dataset_tta, batch_size=batch_size, shuffle=False,
                        worker_init_fn=seed_worker, generator=g, num_workers=2)