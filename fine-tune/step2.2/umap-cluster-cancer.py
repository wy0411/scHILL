import torch
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import os
import seaborn as sns








path = '/path/to/your/data'
os.chdir(path)

group_df = pd.read_csv("./tensors/label.csv",  dtype={"name": str})  
filename_to_group = dict(zip(group_df['name'], group_df['label']))
tensor_list = []
group_list = []
label_list = []


for filename in group_df['name']:
    filename = str(filename)
    file_path = os.path.join("./tensors", filename)
    tensor = torch.load(file_path, map_location=torch.device('cpu'))  
    tensor_list.append(tensor.detach().numpy())
    group_list.append(filename_to_group[filename])
    label_list.append(filename)  

tensor_flattened = [x.flatten() for x in tensor_list]
tensor_array = np.stack(tensor_flattened)

reducer = umap.UMAP(n_components=2, random_state=928, metric = 'cosine')
embedding = reducer.fit_transform(tensor_array)

unique_groups = sorted(set(group_list))
palette = sns.color_palette("hls", len(unique_groups))
group_to_color = {g: palette[i] for i, g in enumerate(unique_groups)}

plt.figure(figsize=(8,6))
for i, (x, y) in enumerate(embedding):
    plt.scatter(x, y, color=group_to_color[group_list[i]], label=group_list[i] if i == group_list.index(group_list[i]) else "", alpha=0.7)
    plt.text(x, y, label_list[i], fontsize=2, ha='right', va='bottom')

handles = []
for group, color in group_to_color.items():
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=group, markersize=8))
plt.legend(handles=handles)
plt.title('UMAP of tensors by group')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.tight_layout()
plt.savefig('./cluster.pdf')
