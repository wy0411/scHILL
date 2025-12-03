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
import pandas as pd

path = '/path/to/your/data'

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
        file_name = self.file_names[idx]
        return tensor, label1, label2, file_name

    def label_to_index(self, label):
        label_dict1 = {'severe/critical': 0, 'mild/moderate': 1, 'control': 2}
        return label_dict1[label]

    def label_to_index2(self, label):
        label_dict2 = {'disease': 0, 'control': 1}
        return label_dict2[label]


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, hidden_size1, hidden_size3, hidden_size5, num_classes):
        super(MLP, self).__init__()
        self.fc0 = nn.Linear(256, 1)

        self.fc1 = nn.Linear(784, hidden_size1)
        self.relu = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.fc2 = nn.Linear(hidden_size1, hidden_size3)
        self.fc3 = nn.Linear(hidden_size3, hidden_size5)
        self.fc7 = nn.Linear(hidden_size5, num_classes)
        self.drop = nn.Dropout(p=0.25)
        self.fc5 = nn.Linear(2, 2)

    def forward(self, x):
        x = (self.fc0(x)).squeeze(2)

        x = self.drop(x)
        out = self.fc1(x)
        out = self.relu(out)

        out = self.drop(out)
        out = self.fc2(out)
        out = self.relu(out)

        out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc7(out)

        # out = self.fc5(out)

        return out


class MLP2(nn.Module):
    def __init__(self, hidden_size1, hidden_size3, hidden_size5):
        super(MLP2, self).__init__()
        self.fc0 = nn.Linear(256, 1)

        self.fc1 = nn.Linear(784, hidden_size1)
        self.relu = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.fc2 = nn.Linear(hidden_size1, hidden_size3)
        self.fc3 = nn.Linear(hidden_size3, hidden_size5)
        self.fc7 = nn.Linear(hidden_size5, 2)
        self.drop = nn.Dropout(p=0.05)
        self.fc5 = nn.Linear(2, 2)

    def forward(self, x):
        x = (self.fc0(x)).squeeze(2)

        x = self.drop(x)
        out = self.fc1(x)
        out = self.relu(out)

        out = self.drop(out)
        out = self.fc2(out)
        out = self.relu(out)

        out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc7(out)

        # out = self.fc5(out)

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


def train_model(model, model2, criterion, optimizer, optimizer2, train_loader, val_loader, num_epochs, device,
                early_stopping, save_path):
    model.to(device)
    model2.to(device)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        model2.train()

        for tensors, labels1, labels2, file_name in train_loader:
            tensors, labels1, labels2 = tensors.to(device), labels1.to(device), labels2.to(device)
            tensors = tensors.squeeze(0)
            # 前向传播
            total_loss1 = 0
            total_loss2 = torch.zeros(1).to(device)

            for i in range(25):
                tensors1 = tensors[i].unsqueeze(0)
                outputs = model(tensors1)
                loss1 = criterion(outputs, labels2)
                if (torch.argmax(outputs, dim=1).item() == 0):
                    outputs = model2(tensors1)
                    new_tensor = torch.zeros(1, 3).to(device)
                    new_tensor[0, 0] = outputs[0, 0]
                    new_tensor[0, 1] = outputs[0, 1]
                    new_tensor[0, 2] = 0
                    loss2 = criterion(new_tensor, labels1).to(device)
                    total_loss2 += loss2
                total_loss1 += loss1
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}',flush=True)
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss1.backward()
            optimizer.step()

            if (total_loss2.item() != 0):
                optimizer2.zero_grad()
                total_loss2.backward()
                optimizer2.step()

        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss2.item():.4f}', flush=True)
        # 验证阶段
        model.eval()
        model2.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for tensors, labels1, labels2, file_name in val_loader:
                tensors, labels1, labels2 = tensors.to(device), labels1.to(device), labels2.to(device)
                tensors1 = tensors.squeeze(0)
                outputs = torch.nn.functional.softmax(model(tensors1), dim=1).mean(dim=0).unsqueeze(0)

                if (torch.argmax(outputs, dim=1).item() == 0):  # disease
                    outputs = torch.nn.functional.softmax(model2(tensors1), dim=1).mean(dim=0).unsqueeze(0)
                    new_tensor = torch.zeros(1, 3).to(device)
                    new_tensor[0, 0] = outputs[0, 0]
                    new_tensor[0, 1] = outputs[0, 1]
                    new_tensor[0, 2] = 0
                    _, preds = torch.max(new_tensor, 1)
                    val_labels.extend(labels1.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())


                elif (torch.argmax(outputs, dim=1).item() == 1):  # control
                    new_tensor = torch.zeros(1, 3).to(device)
                    new_tensor[0, 0] = outputs[0, 0] / 2
                    new_tensor[0, 1] = outputs[0, 0] / 2
                    new_tensor[0, 2] = outputs[0, 1]
                    _, preds = torch.max(new_tensor, 1)
                    val_labels.extend(labels1.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        best_accuracy, best_precision, best_recall, best_f1, best_mcc = early_stopping(val_accuracy, val_precision,
                                                                                       val_recall, val_f1, val_mcc)

        if early_stopping.best_score == val_mcc:
            torch.save({
                'model_state_dict': model.state_dict(),
                'model2_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),
                'epoch': epoch,
                'val_mcc': val_mcc
            }, save_path)

        if early_stopping.early_stop:
            # print("Early stopping triggered. Stopping training...")
            break

    return best_accuracy, best_precision, best_recall, best_f1, best_mcc


def main(random_state=777):
    os.chdir(path)
    tensor_dir = './tensors'
    labels_file = './tensors/label.csv' #label file
    for i in range(1, 6):
        os.makedirs(os.path.join(path, str(i)), exist_ok=True)
    dataset = TensorDataset(tensor_dir, labels_file)

    hidden_size1 = 784
    hidden_size3 = 784
    hidden_size5 = 784
    num_classes = 2
    num_epochs = 100
    learning_rate = 0.001

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for i in range(1, 6):
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        all_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
            print(f"Starting Fold {fold + 1}/5", flush=True)
            model = MLP(hidden_size1, hidden_size3, hidden_size5, num_classes).to(device)
            model2 = MLP2(hidden_size1, hidden_size3, hidden_size5).to(device)
            early_stopping = EarlyStopping(patience=150, delta=0, verbose=False)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
            optimizer2 = optim.AdamW(model2.parameters(), lr=learning_rate / 5, betas=(0.9, 0.999))

            train_data = torch.utils.data.Subset(dataset, train_idx)
            val_data = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
            save_model_path = f'./{i}/best_model_fold{fold}.pt'
            val_accuracy, val_precision, val_recall, val_f1, val_mcc = train_model(model, model2, criterion, optimizer,
                                                                               optimizer2, train_loader, val_loader,
                                                                               num_epochs, device, early_stopping,
                                                                               save_model_path)
            model = MLP(hidden_size1, hidden_size3, hidden_size5, num_classes).to(device)
            model2 = MLP2(hidden_size1, hidden_size3, hidden_size5).to(device)

            checkpoint = torch.load(f'./{i}/best_model_fold{fold}.pt', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model2.load_state_dict(checkpoint['model2_state_dict'])

            with torch.no_grad():
                for tensors, labels1, labels2, file_name in val_loader:
                    tensors, labels1, labels2, file_name = tensors.to(device), labels1.to(device), labels2.to(
                        device), file_name
                    tensors1 = tensors.squeeze(0)
                    outputs = torch.nn.functional.softmax(model(tensors1), dim=1).mean(dim=0).unsqueeze(0)
                    if (torch.argmax(outputs, dim=1).item() == 0):  # disease
                        outputs = torch.nn.functional.softmax(model2(tensors1), dim=1).mean(dim=0).unsqueeze(0)
                        if labels2 == 0:
                            with open(f"{i}/{i}.log", "a") as f:
                                f.write(f'file:{file_name}, point:{outputs[0, 1].item() + outputs[0, 0].item() * 2}\n')

    folders = ["1", "2", "3", "4", "5"]
    records = []
    for folder in folders:
        log_file = os.path.join(path, folder, f"{folder}.log")
        with open(log_file, "r") as f:
            for line in f:
                if "point" in line:   # 只保留含 point 的行
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
    main(random_state=555)
