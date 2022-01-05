from pathlib import Path
import pytorch_lightning as pl
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.strategy import NegativeSampling
from openke.module.loss import MarginLoss
from openke.module.model import TransE
from openke.config import Trainer, Tester
import openke
from clearml import Task, StorageManager, Dataset
import argparse
import json
import ipdb
import os
import pandas as pd

task = Task.init(project_name='gdelt-embeddings', task_name='graph-training',
                 output_uri="s3://experiment-logging/storage/")
#task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04", docker_setup_bash_script=['git clone https://github.com/thunlp/OpenKE','cd OpenKE/openke', 'rm -r release', 'bash make.sh'])
task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")

# task.connect(args)
task.execute_remotely(queue_name="compute2", exit_process=True)
#clearlogger = task.get_logger()

dataset_obj = Dataset.get(dataset_project="datasets/gdelt",
                          dataset_name="gdelt_openke_format", only_published=True)
dataset_path = dataset_obj.get_local_copy()
print(list(os.walk(dataset_path)))


def read_file(dataset_path: str, file_name: str):
    data = pd.read_csv('{}/{}'.format(dataset_path, file_name),
                       sep="\t", header=None, dtype=str)[1:].reset_index(drop=True)
    return data


train_data = read_file(dataset_path, "train2id.txt")


# dataloader for training
train_dataloader = TrainDataLoader(
    in_path=dataset_path+"/",
    nbatches=128,  # number of batches
    threads=4,
    sampling_mode="normal",
    bern_flag=1,  # noise for negative triples
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)

# define the model
transe = TransE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200,  # embedding dim
    p_norm=1,
    norm_flag=True)

# dataloader for test
#test_dataloader = TestDataLoader(dataset_path)

# define the loss function
model = NegativeSampling(
    model=transe,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)

print(train_dataloader.get_batch_size())

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader,
                  train_times=1000, alpha=0.01, use_gpu=True)
trainer.run()

Path('./checkpoint/').mkdir(parents=True, exist_ok=True)
transe.save_checkpoint('./checkpoint/transe.ckpt')

task.upload_artifact(name='transe.ckpt', artifact_object='./checkpoint/transe.ckpt')

