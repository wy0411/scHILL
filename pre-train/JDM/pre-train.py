import torch
import torch.nn as nn
import anndata
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import model_mae
import os
import random
import torch.nn.init as init
import pandas as pd
from statistics import mean

path = '/your/path/to/files' #your own path

class mae2(nn.Module):
    def __init__(self):
        super(mae2, self).__init__()
        self.norm_layer = torch.nn.LayerNorm(224)
        self.vit_mae = model_mae.mae_vit_base_patch16_dec512d8b()

    def forward(self, x):
        x_norm = self.norm_layer(x)
        y = x_norm.unsqueeze(1)  # Add channel dimension
        loss_mae, y, mask = self.vit_mae(y)
        y = y.squeeze(1)  # Delete channel dimension

        return y, loss_mae


def random_crop(matrix, crop_shape):
    original_shape = matrix.shape
    target_rows, target_cols = crop_shape

    if original_shape[0] < target_rows or original_shape[1] < target_cols:
        raise ValueError("Crop shape is larger than the original matrix dimensions.")

    start_row = np.random.randint(0, original_shape[0] - target_rows + 1)
    start_col = np.random.randint(0, original_shape[1] - target_cols + 1)

    cropped_matrix = matrix[start_row:start_row + target_rows, start_col:start_col + target_cols]
    return cropped_matrix


def resize_matrix(matrix, target_shape=(224, 224)):
    target_rows, target_cols = target_shape
    rows, cols = matrix.shape

    # Crop the matrix if it's larger than the target shape
    if rows > target_rows or cols > target_cols:
        matrix = random_crop(matrix, target_shape)

    # Pad the matrix if it's smaller than the target shape
    if rows < target_rows or cols < target_cols:
        new_matrix = np.zeros(target_shape)
        new_matrix[:rows, :cols] = matrix
        matrix = new_matrix

    return matrix


def load_data_from_h5ad(file_path):
    adata = anndata.read_h5ad(file_path)
    # if adata.raw is None:
    #    print(f"Skipping {file_path}: `adata.raw` is missing.", flush=True)
    #    return None, None, None
    num_cells = adata.n_obs
    num_genes = adata.n_vars
    expression_matrix = adata.X
    if not isinstance(expression_matrix, np.ndarray):
        expression_matrix = expression_matrix.toarray()  # Convert sparse matrix to dense
    col_means = expression_matrix.mean(axis=0)

    sorted_col_indices = np.argsort(col_means)[-500:][::-1]
    matrix_sorted_by_col = expression_matrix[:, sorted_col_indices]

    sorting_keys = tuple(-matrix_sorted_by_col[:, i] for i in range(99, -1, -1))
    sorted_row_indices = np.lexsort(sorting_keys)
    sorted_matrix = matrix_sorted_by_col[sorted_row_indices]

    return sorted_matrix, num_cells, num_genes


def get_random_file_from_folder(folder_path):
    files = os.listdir(folder_path)
    h5ad_files = [f for f in files if f.endswith('.h5ad')]
    if not h5ad_files:
        raise FileNotFoundError("No .h5ad files found in the specified folder.")
    return os.path.join(folder_path, random.choice(h5ad_files))


def matrix2tensor(matrix):
    tensor = torch.from_numpy(matrix).float().unsqueeze(0)
    dataset = TensorDataset(tensor, tensor)
    return dataset

flag = 0
previousmin = 100

def train(model_mae2, data_loader, learning_rate, device):
    criterion = nn.MSELoss()

    optimizer_mae2 = torch.optim.Adam(model_mae2.parameters(), lr=learning_rate)

    model_mae2.to(device)
    model_mae2.train()

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        tr = inputs.clone()

        mae_outputs2, loss_mae2 = model_mae2(tr)

        optimizer_mae2.zero_grad()
        loss_mae2.backward()
        optimizer_mae2.step()

        print(f"MAE Model loss:{loss_mae2.item()}", flush=True)


# Assuming you have a DataLoader `data_loader` with your data
if __name__ == "__main__":
    os.chdir(path)
    folder_path = "./h5ad" # Folder stores h5ad files. If your h5ad file is merged and contains numerous individuals, please split it.
    model_save_path2 = "./pretrain/mae.pth" # Folder stores pre-trained models.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mae2 = mae2()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(10001):
        file_path = get_random_file_from_folder(folder_path)
        expression_matrix, num_cells, num_genes = load_data_from_h5ad(file_path)

        expression_matrix = resize_matrix(expression_matrix)

        # Convert expression_matrix to Tensor and create a DataLoader
        dataset = matrix2tensor(expression_matrix)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Train model
        train(model_mae2, data_loader, learning_rate=1e-3, device=device)
        print(f'Epoch {epoch + 1}')
        if (epoch + 1 > 499) and ((epoch + 1) % 500 == 0):
            path2 = f"./pretrain/model_mae_epoch_{epoch + 1}.pth"

            torch.save({'model_mae2_state_dict': model_mae2.state_dict()}, path2)

    # Save model parameters after all epochs are done

    torch.save({'model_mae2_state_dict': model_mae2.state_dict()}, model_save_path2)
