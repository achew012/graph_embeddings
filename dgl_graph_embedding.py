from clearml import Task, StorageManager, Dataset
import os
import pandas as pd
import numpy as np
import tqdm
import ipdb

#remote_path = "s3://experiment-logging"
#task = Task.init(project_name='gdelt-embeddings', task_name='dgl-graph-training',
#                 output_uri=os.path.join(remote_path, "storage"))
#task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
#task.execute_remotely(queue_name="compute2", exit_process=True)

dataset_obj = Dataset.get(dataset_project="datasets/gdelt", dataset_name="gdelt_openke_format_w_extras", only_published=True)
dataset_path = dataset_obj.get_local_copy()

def read_file(dataset_path:str, file_name:str):
    data = pd.read_csv('{}/{}'.format(dataset_path, file_name), sep="\t", header=None, dtype=str)[1:].reset_index(drop=True)
    return data

entity_dict = read_file(dataset_path, "entity2id.txt")
relation_dict = read_file(dataset_path, "relation2id.txt")
train_data = read_file(dataset_path, "train2id.txt")
train_data[5] = train_data[4].str[:6]

from dgl.data import dgl_dataset
import pytorch_lightning as pl

from dgl.data import GDELTDataset

train_data = GDELTDataset()
valid_data = GDELTDataset(mode='valid')
test_data = GDELTDataset(mode='test')

ipdb.set_trace()

from dgl.nn.pytorch.link.transe import TransE

