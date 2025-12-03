import torch
import torch.nn as nn
import anndata
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import model_mae_finetune
import os
import scanpy as sc
import numpy as np
import random
import torch.nn.init as init
import pandas as pd

#If you want to use specific genes, you should modify line 124. 
#If you want to use specific size, you shoule modify line 67, 68, 77, 94, 95, 124, 134
path = '/path/to/your/data'
class mae2(nn.Module):
    def __init__(self):
        super(mae2, self).__init__()
        self.norm_layer = torch.nn.LayerNorm(224)
        self.vit_mae = model_mae_finetune.mae_vit_base_patch16()

    def forward(self, x):
        x_norm = self.norm_layer(x)
        x = x.unsqueeze(1)  # Add channel dimension
        latent = self.vit_mae(x)
        x = latent.squeeze(1)  # Delete channel dimension

        return x


def quality_control_and_normalize(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def load_and_process_data(file_path):
    adata = anndata.read_h5ad(file_path)
    adata = quality_control_and_normalize(adata)
    expression_matrix = adata.X
    if not isinstance(expression_matrix, np.ndarray):
        expression_matrix = expression_matrix.toarray()

    return adata, expression_matrix

def filter_genes(adata, expression_matrix, gene_list):
    gene_indices = {gene: idx for idx, gene in enumerate(adata.var_names)}
    filtered_matrix = np.zeros((expression_matrix.shape[0], len(gene_list)))

    for i, gene in enumerate(gene_list):
        if gene in gene_indices:
            filtered_matrix[:, i] = expression_matrix[:, gene_indices[gene]]

    return filtered_matrix


def sort_and_select_genes(adata, expression_matrix, gene_list):
    sorted_genes = sorted(gene_list)
    expression_matrix = filter_genes(adata, expression_matrix, sorted_genes)

    col_means = np.var(expression_matrix, axis=0)
    sorted_col_indices = np.argsort(-col_means)
    selected_matrix = expression_matrix[:, sorted_col_indices[:224]]
    selected_genes = [sorted_genes[i] for i in sorted_col_indices[:224]]

    return selected_matrix, selected_genes


def sort_and_select_cells(expression_matrix):
    col_means = np.mean(expression_matrix, axis=1)
    sorted_col_indices = np.argsort(col_means)

    return expression_matrix[sorted_col_indices[:224], :]


def get_final_matrix(file_path, gene_list_path):
    adata, expression_matrix = load_and_process_data(file_path)
    if adata is None:
        return None

    with open(gene_list_path, 'r') as f:
        gene_list = f.read().splitlines()
    expression_matrix = sort_and_select_cells(expression_matrix)
    expression_matrix, selected_genes = sort_and_select_genes(adata, expression_matrix, gene_list)

    return expression_matrix, selected_genes


def matrix2tensor(matrix):
    row_split = np.split(matrix, 1, axis=0) #average split for row (cells)
    matrix2 = np.array([np.split(sub_matrix, 1, axis=1) for sub_matrix in row_split]).reshape(1, 224, 224) #average split for column (genes)
    tensor = torch.from_numpy(matrix2).float()
    dataset = TensorDataset(tensor, tensor)
    return dataset


def train(model_mae2, data_loader, device):
    model_mae2.to(device)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        tr = inputs.clone()
        mae_outputs2 = model_mae2(tr)
        final_output = mae_outputs2
        return final_output


if __name__ == "__main__":
    os.chdir(path)
    single_cell_model_path = './pretrain/model_mae_epoch_10000.pth' # pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_mae2 = mae2()
    single_cell_checkpoint = torch.load(single_cell_model_path)
    model_mae2.load_state_dict(single_cell_checkpoint['model_mae2_state_dict'])

    input_folder = './h5ad'
    output_folder = './tensors'
    gene_list_path = './hvg' # the hvg file
    os.makedirs(output_folder, exist_ok=True)
    all_selected_genes = set()
    # 遍历文件夹中所有 .h5ad 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".h5ad"):
            file_path = os.path.join(input_folder, filename)
            expression_matrix, selected_genes = get_final_matrix(file_path, gene_list_path)
            #all_selected_genes.update(selected_genes)
            dataset = matrix2tensor(expression_matrix)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False) #batchsize X should be the same with N ,where (N,224,224) in line 95
            output_tensor = train(model_mae2, data_loader, device=device)
            output_filename = os.path.splitext(filename)[0] + ".pt"
            output_file_path = os.path.join(output_folder, output_filename)
            torch.save(output_tensor, output_file_path)
