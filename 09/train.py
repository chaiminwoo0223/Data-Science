from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
import torch

device = torch.device('cuda:0')

def train_val(model, epochs, loader_train, loader_valid, optimizer, criterion, scheduler):
    for epoch in range(epochs):
        # 훈련
        model.train()
        epoch_train_loss = 0
        for images, labels in tqdm(loader_train):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, labels)
            epoch_train_loss += loss.item() 
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'에포크 [{epoch+1}/{epochs}] - 훈련 데이터 손실값: {epoch_train_loss/len(loader_train):.4f}')
    
        # 검증
        model.eval()
        epoch_valid_loss = 0
        preds_list = []
        true_onehot_list = []
        with torch.no_grad():
            for images, labels in loader_valid:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_valid_loss += loss.item()
                preds = torch.softmax(outputs.cpu(), dim=1).numpy()
                true_onehot = torch.eye(4)[labels.cpu()].numpy()
                preds_list.extend(preds)
                true_onehot_list.extend(true_onehot)
        print(f'에포크 [{epoch+1}/{epochs}] - 검증 데이터 손실값: {epoch_valid_loss/len(loader_valid):.4f}')
        print(f'검증 데이터 ROC AUC: {roc_auc_score(true_onehot_list, preds_list):.4f}')
    return model