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
from sklearn.model_selection import KFold
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import re

path = '/path/to/your/data'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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
        name = self.file_names[idx]
        return tensor, label1, label2, name

    def label_to_index(self, label):
        label_dict1 = {'C': 0, 'JD': 1}
        return label_dict1[label]

    def label_to_index2(self, label):
        label_dict2 = {'a': 0}
        return label_dict2[label]


class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_mcc = -float("inf")  # 最佳 MCC 值
        self.accuracy_best = -float("inf")
        self.precision_best = -float("inf")
        self.recall_best = -float("inf")
        self.f1_best = -float("inf")

    def __call__(self, val_accuracy, val_precision, val_recall, val_f1, val_mcc):
        score = val_mcc
        accuracy = val_accuracy
        precision = val_precision
        recall = val_recall
        f1 = val_f1

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
            self.accuracy_best = accuracy
            self.precision_best = precision
            self.recall_best = recall
            self.f1_best = f1
            self.counter = 0
        return self.accuracy_best, self.precision_best, self.recall_best, self.f1_best, self.best_score


class MLP2(nn.Module):
    def __init__(self, hidden_size1, hidden_size3, hidden_size5):
        super(MLP2, self).__init__()
        self.fc0 = nn.Linear(64, 1)

        self.fc1 = nn.Linear(64, 2)
        self.relu = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.fc2 = nn.Linear(64, 3)
        # self.fc2 = nn.Linear(64,2)
        self.fc3 = nn.Linear(64, 3)
        self.fc7 = nn.Linear(64, 3)
        self.drop = nn.Dropout(p=0.25)
        self.fc5 = nn.Linear(2, 2)

    def forward(self, x):
        x = (self.fc0(x)).squeeze(2)
        out = self.fc1(x)
        return out


def seed_torch(seed=777):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_model(model2, criterion, optimizer2, train_loader, val_loader, num_epochs, device, early_stopping, save_path):
    model2.to(device)

    for epoch in range(num_epochs):
        # 训练阶段

        model2.train()

        for tensors, labels1, labels2, names in train_loader:
            tensors, labels1, labels2 = tensors.to(device), labels1.to(device), labels2.to(device)
            tensors = tensors.squeeze(0)
            # 前向传播
            total_loss1 = 0
            total_loss2 = torch.zeros(1).to(device)
            outputs = model2(tensors)
            preds = torch.max(outputs, 1)
            total_loss2 = criterion(outputs, labels1).to(device)
            optimizer2.zero_grad()
            total_loss2.backward()
            optimizer2.step()
        model2.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for tensors, labels1, labels2, file_name in val_loader:
                tensors, labels1, labels2, file_name = tensors.to(device), labels1.to(device), labels2.to(
                    device), file_name
                tensors1 = tensors.squeeze(0)
                outputs = torch.nn.functional.softmax(model2(tensors1), dim=1)
                _, preds = torch.max(outputs, 1)
                val_labels.extend(labels1.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        val_accuracy, val_precision, val_recall, val_f1, best_mcc = early_stopping(val_accuracy, val_precision,
                                                                                   val_recall, val_f1, val_mcc)
        if val_mcc == best_mcc:
            torch.save({'model2_state_dict': model2.state_dict(), 'optimizer2_state_dict': optimizer2.state_dict()},
                       save_path)

    return val_accuracy, val_precision, val_recall, val_f1, val_mcc


def main():
    os.chdir(path)
    tensor_dir = './tensors'
    labels_file = './tensors/label.csv' #label file
    dataset = TensorDataset(tensor_dir, labels_file)
    input_size = 784 * 256
    hidden_size1 = 784
    hidden_size3 = 784
    hidden_size5 = 784
    num_classes = 2
    num_epochs = 500
    learning_rate = 0.5
    for i in range(1, 6):
        os.makedirs(os.path.join(path, str(i)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(seed=1)
    label = dataset.labels1
    for i in range(1, 6):
        kf = KFold(n_splits=10, shuffle=True, random_state=i)
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
            print(f"Starting Fold {fold + 1}/5", flush=True)
            model2 = MLP2(hidden_size1, hidden_size3, hidden_size5).to(device)
            early_stopping = EarlyStopping(patience=150, delta=0, verbose=False)
            criterion = nn.CrossEntropyLoss()
            optimizer2 = optim.AdamW(model2.parameters(), lr=learning_rate, betas=(0.9, 0.999))

            train_data = torch.utils.data.Subset(dataset, train_idx)
            val_data = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
            save_model_path = f'./{i}/best_model_fold{fold}.pt'
            val_accuracy, val_precision, val_recall, val_f1, val_mcc = train_model(model2, criterion,
                                                                               optimizer2, train_loader, val_loader,
                                                                               num_epochs, device, early_stopping,
                                                                               save_model_path)
            model2 = MLP2(hidden_size1, hidden_size3, hidden_size5).to(device)

            checkpoint = torch.load(f'./{i}/best_model_fold{fold}.pt',
                                weights_only=False)
            model2.load_state_dict(checkpoint['model2_state_dict'])

            with torch.no_grad():
                for tensors, labels1, labels2, file_name in val_loader:
                    tensors, labels1, labels2, file_name = tensors.to(device), labels1.to(device), labels2.to(
                        device), file_name
                    tensors1 = tensors.squeeze(0)
                    outputs = torch.nn.functional.softmax(model2(tensors1), dim=1)
                    if labels1 == 1: #disease
                        print(f'file:{file_name}, point:{outputs[0, 1].item()}')

    folders = ["1", "2", "3", "4", "5"]
    records = []
    for folder in folders:
        log_file = os.path.join(path, folder, f"{folder}.log")
        with open(log_file, "r") as f:
            for line in f:
                if "point" in line:  # 只保留含 point 的行
                    # 删除无用字符，保留 样本名 和 分数
                    clean = re.sub(r"file:\('", "", line)
                    clean = re.sub(r"',\), point:\s*", "\t", clean)
                    sample, score = clean.strip().split("\t")
                    records.append([sample, float(score)])

    # 转为 DataFrame
    df = pd.DataFrame(records, columns=["sample", "score"])

    # 统计次数 & 均值
    result = df.groupby("sample").agg(
        avg_score=("score", "mean")
    ).reset_index()

    # 输出
    output_file = "point.tsv"
    result.to_csv(output_file, sep="\t", index=False)


if __name__ == '__main__':
    main()
