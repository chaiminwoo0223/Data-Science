from efficientnet_pytorch import EfficientNet
from transformers import get_cosine_schedule_with_warmup
from label_smoothing import apply_label_smoothing
from data import test, submission
from dataloader import loader_train, loader_valid, loader_test, loader_tta
from train import train_val
import torch
import torch.nn as nn
import random
import numpy as np
import os

# 시드값 고정
seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# GPU
device = torch.device('cuda:0')

# 하이퍼파라미터
batch_size = 32
epochs = 3
alpha = 0.001
threshold = 0.999
target = ['healthy', 'multiple_diseases', 'rust', 'scab']

# 모델 생성
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
model = model.to(device)

# 손실함수, 옵티마이저, 스케줄러
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006, weight_decay=0.0001)
scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=len(loader_train)*5,
                                            num_training_steps=len(loader_train)*epochs)

# 모델 훈련 및 성능 검증
train_val(model, epochs, loader_train, loader_valid, optimizer, criterion, scheduler)

# 예측(1)
model.eval()
preds = np.zeros((len(test), 4))
with torch.no_grad():
    for i, images in enumerate(loader_test):
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
        preds[i*batch_size:(i+1)*batch_size] += preds_part

# 결과 제출(1)
submission_test = submission.copy()
submission_test[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds
submission_test.to_csv('submission_test.csv', index=False)
submission_test_ls = submission_test.copy()
submission_test_ls[target] = apply_label_smoothing(submission_test_ls, target, alpha, threshold)
submission_test_ls.to_csv('submission_test_ls.csv', index=False)

# 예측(2)
num_tta = 5
preds_tta = np.zeros((len(test), 4))
for _ in range(num_tta):
    with torch.no_grad():
        for i, images in enumerate(loader_tta):
            images = images.to(device)
            outputs = model(images)
            preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
            preds_tta[i*batch_size:(i+1)*batch_size] += preds_part
preds_tta /= num_tta

# 결과 제출(2)
submission_tta = submission.copy()
submission_tta[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_tta
submission_tta.to_csv('submission_tta.csv', index=False)
submission_tta_ls = submission_tta.copy()
submission_tta_ls[target] = apply_label_smoothing(submission_tta_ls, target, alpha, threshold)
submission_tta_ls.to_csv('submission_tta_ls.csv', index=False)