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

path = '/path/to/your/data'

class ProjectionModel(nn.Module):
    def __init__(self, num_cells, num_genes):
        super(ProjectionModel, self).__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        # MLP for initial dimensionality reduction
        self.mlp1 = nn.Linear(num_genes, 448)
        self.mlp2 = nn.Linear(num_cells, 448)
        self.loss = nn.MSELoss()

    def forward(self, input):
        x = input + 1
        x = self.mlp1(x)
        m = x.clone()
        q = m.permute(0, 2, 1)
        qq = q.clone()
        qqq = nn.functional.linear(qq, self.mlp2.weight.clone(), self.mlp2.bias)
        e = qqq.clone()
        w = e.permute(0, 2, 1)

        y = w.clone()
        t = y.permute(0, 2, 1)
        r = t.clone()
        r = r - self.mlp2.bias
        pinv1 = torch.linalg.pinv(self.mlp2.weight)
        r = torch.matmul(r, pinv1.T)
        u = r.clone()
        u = u.permute(0, 2, 1)
        g = u.clone()
        g = g - self.mlp1.bias
        pinv2 = torch.linalg.pinv(self.mlp1.weight)
        g = torch.matmul(g, pinv2.T)
        g = g - 1

        projection_loss = self.loss(g, input)
        out = w.clone()

        return out, projection_loss


class mae2(nn.Module):
    def __init__(self):
        super(mae2, self).__init__()
        self.norm_layer = torch.nn.LayerNorm(448)
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


def resize_matrix(matrix, target_shape=(4000, 10000)):
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
    if adata.raw is None:
        print(f"Skipping {file_path}: `adata.raw` is missing.", flush=True)
        return None, None, None
    num_cells = adata.n_obs
    num_genes = adata.n_vars
    expression_matrix = adata.raw.to_adata().X
    if not isinstance(expression_matrix, np.ndarray):
        expression_matrix = expression_matrix.toarray()  # Convert sparse matrix to dense
    col_means = expression_matrix.mean(axis=0)

    sorted_col_indices = np.argsort(col_means)[-10000:][::-1]
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
    row_split = np.split(matrix, 2, axis=0)
    matrix2 = np.array([np.split(sub_matrix, 2, axis=1) for sub_matrix in row_split]).reshape(4, 2000, 5000)
    tensor = torch.from_numpy(matrix2).float()
    dataset = TensorDataset(tensor, tensor)
    return dataset

flag = 0
previousmin = 100

# renew = 0
def train(model_mae2, model_projection1a,  model_projection1c, model_projection3a,
          data_loader, learning_rate,
          device):
    criterion = nn.MSELoss()
    optimizer_mae2 = torch.optim.Adam(model_mae2.parameters(), lr=learning_rate)
    optimizer_projection1a = torch.optim.Adam(model_projection1a.parameters(), lr=2e-4)
    optimizer_projection1c = torch.optim.Adam(model_projection1c.parameters(), lr=2e-4)
    optimizer_projection3a = torch.optim.Adam(model_projection3a.parameters(), lr=2e-4)

    model_mae2.to(device)
    model_mae2.train()

    model_projection1a.to(device)
    model_projection1c.to(device)
    model_projection3a.to(device)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        tr = inputs.clone()

        split_tensors = torch.chunk(tr, chunks=4, dim=0)
        matrix1 = split_tensors[0]
        matrix1a = matrix1[:, :1000, :2500]
        matrix1c = matrix1[:, 1000:, :2500]
        matrix3 = split_tensors[2]
        matrix3a = matrix3[:, :, :2500]

        projected1a, loss1a = model_projection1a(matrix1a)
        optimizer_projection1a.zero_grad()
        loss1a.backward()
        optimizer_projection1a.step()

        projected1c, loss1c = model_projection1c(matrix1c)
        optimizer_projection1c.zero_grad()
        loss1c.backward()
        optimizer_projection1c.step()

        projected3a, loss3a = model_projection3a(matrix3a)
        optimizer_projection3a.zero_grad()
        loss3a.backward()
        optimizer_projection3a.step()

        coeffs = torch.rand(17, 3)
        coeffs /= coeffs.sum(dim=1, keepdim=True)
        new_tensors = torch.cat(
            [(alpha * projected1a + beta * projected1c + gamma * projected3a) for alpha, beta, gamma in coeffs], dim=0)

        projected2 = torch.cat([projected1a, projected1c, projected3a, new_tensors], dim=0)
        pro2 = projected2.clone()
        mae_outputs2, loss_mae2 = model_mae2(pro2)


        optimizer_mae2.zero_grad()
        loss_mae2.backward()
        optimizer_mae2.step()

        print(f"MAE Model loss: {loss_mae2.item()}", flush=True)


# Assuming you have a DataLoader `data_loader` with your data
if __name__ == "__main__":
    folder_path = "./h5ad" # Folder stores h5ad files. If your h5ad file is merged and contains numerous individuals, please split it.
    model_save_path2 = "./pretrain/mae.pth" # Folder stores pre-trained models.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_projection1a = ProjectionModel(num_cells=1000, num_genes=2500)
    model_projection1c = ProjectionModel(num_cells=1000, num_genes=2500)
    model_projection3a = ProjectionModel(num_cells=2000, num_genes=2500)
    model_mae2 = mae2()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(60000):
        file_path = get_random_file_from_folder(folder_path)
        expression_matrix, num_cells, num_genes = load_data_from_h5ad(file_path)
        expression_matrix = resize_matrix(expression_matrix)

        # Convert expression_matrix to Tensor and create a DataLoader
        dataset = matrix2tensor(expression_matrix)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Train model
        train( model_mae2, model_projection1a, model_projection1c, model_projection3a,
               data_loader,
              learning_rate=6e-4, device=device)
        print(f'Epoch {epoch + 1}')
        if (epoch + 1 > 499) and ((epoch + 1) % 500 == 0):
            path2 = f"./pretrain/model_mae_epoch_{epoch + 1}.pth"

            torch.save({'model_mae2_state_dict': model_mae2.state_dict()}, path2)

    # Save model parameters after all epochs are done

    torch.save({'model_mae2_state_dict': model_mae2.state_dict()}, model_save_path2)
