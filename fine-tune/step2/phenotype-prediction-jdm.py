import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import classification_report
from timm.models.vision_transformer import PatchEmbed, Block





path='/path/to/your/data'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def seed_torch(seed=777):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

# 定义数据集类
class TensorDataset(Dataset):
    def __init__(self, tensor_dir, labels_file):
        self.tensor_dir = tensor_dir

        self.labels = pd.read_csv(labels_file)
        self.file_names = self.labels['name']
        self.labels1 = self.labels['label'].apply(self.label_to_index).values
        self.labels2 = self.labels['label2'].apply(self.label_to_index2).values

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, self.file_names[idx] + '.pt')
        tensor = torch.load(tensor_path)
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]
        return tensor, label1, label2

    def label_to_index(self, label):
        label_dict1 = {'JD':0, 'C':1}
        return label_dict1[label]

    def label_to_index2(self, label):
        label_dict2 = {'a': 0}
        return label_dict2[label]




# 定义MLP模型

class MLP2(nn.Module):
    def __init__(self,  hidden_size1, hidden_size3, hidden_size5):
        super(MLP2, self).__init__()
        self.fc0 = nn.Linear(64, 1)

        self.fc1 = nn.Linear(64, 2)
        self.relu = torch.nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x):
        x = (self.fc0(x)).squeeze(2)

        #x = self.drop(x)
        out = self.fc1(x)
        out = self.relu(out)

        return out

class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_mcc = -float("inf")  # 最佳 MCC 值

    def __call__(self, val_mcc):
        score = val_mcc

        # 初始化最佳分数
        if self.best_score is None:
            self.best_score = score
        # 如果当前分数没有改善
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # 如果当前分数有改善
        else:
            self.best_score = score
            self.counter = 0

def train_model(model2, criterion, optimizer2, train_loader, val_loader, test_loader, num_epochs, device, early_stopping):
    model2.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model2.train()

        for tensors, labels1, labels2 in train_loader:
            tensors, labels1, labels2 = tensors.to(device), labels1.to(device), labels2.to(device)
            tensors = tensors.squeeze(0)
            # 前向传播
            outputs = model2(tensors)
            total_loss2 = criterion(outputs, labels1).to(device)
            optimizer2.zero_grad()
            total_loss2.backward()
            optimizer2.step()
            
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss2.item():.4f}', flush=True)      
        # Validation
        model2.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for tensors, labels1, labels2 in val_loader:
                tensors, labels1, labels2 = tensors.to(device), labels1.to(device), labels2.to(device)
                tensors = tensors.squeeze(0)
                #outputs = torch.nn.functional.softmax(model2(tensors1),dim=1).mean(dim=0).unsqueeze(0)
                outputs = model2(tensors)
                _, preds = torch.max(outputs, 1)
                val_labels.extend(labels1.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, '
              f'Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, MCC: {val_mcc:.4f}', flush=True)
        print(classification_report(val_labels, val_preds, target_names=['JDM', 'Control']))

        early_stopping(val_mcc)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training...")
            break

        val_labels = []
        val_preds = []
        #Test
        for tensors, labels1, labels2 in test_loader:
                tensors, labels1, labels2 = tensors.to(device), labels1.to(device), labels2.to(device)
                tensors = tensors.squeeze(0)
                #outputs = torch.nn.functional.softmax(model2(tensors1),dim=1).mean(dim=0).unsqueeze(0)
                outputs = model2(tensors)
                _, preds = torch.max(outputs, 1)
                val_labels.extend(labels1.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, '
              f'Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, MCC: {val_mcc:.4f}', flush=True)
        print(classification_report(val_labels, val_preds, target_names=['JDM', 'Control']))

def main(random_state=777):
    os.chdir(path)
    tensor_val_dir = './tensors'
    labels_val_dir = './tensors/label.csv' #label file
    dataset_val = TensorDataset(tensor_val_dir, labels_val_dir)
    labels1 = dataset_val.labels1
    train_data, val_data = train_test_split(dataset_val, test_size=0.4, random_state=random_state,stratify=labels1)
    labels2 = [item[1] for item in val_data]
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=random_state, stratify=labels2)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    hidden_size1 = 64
    hidden_size3 = 64
    hidden_size5 = 64
    num_classes = 2
    num_epochs = 500
    learning_rate = 5e-2

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    model2 = MLP2(hidden_size1, hidden_size3, hidden_size5).to(device)
    early_stopping = EarlyStopping(patience=50, delta=0, verbose=False)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(model.parameters(), lr= learning_rate , betas=(0.9, 0.999))
    optimizer2 = optim.AdamW(model2.parameters(), lr= learning_rate/5 , betas=(0.9, 0.999))
    train_model(model2,criterion, optimizer2, train_loader, val_loader, test_loader, num_epochs, device, early_stopping)


if __name__ == '__main__':
    seed_torch(777)
    main(random_state=1)
