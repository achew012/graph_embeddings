from pathlib import Path
import pytorch_lightning as pl

from openke.data import TrainDataLoader, TestDataLoader, PyTorchTrainDataLoader
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
import numpy as np

args = {
    "nbatches": 100,
    "nepochs": 500,
    "lr": 0.01
}

task = Task.init(project_name='gdelt-embeddings', task_name='openke-graph-training',
                 output_uri="s3://experiment-logging/storage/")
task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
task.connect(args)
task.execute_remotely(queue_name="compute2", exit_process=True)
clearlogger = task.get_logger()

dataset_obj = Dataset.get(dataset_project="datasets/gdelt",
                          dataset_name="gdelt_openke_format_w_extras", only_published=True)
dataset_path = dataset_obj.get_local_copy()
print(list(os.walk(dataset_path)))

def read_file(dataset_path: str, file_name: str):
    data = pd.read_csv('{}/{}'.format(dataset_path, file_name),
                       sep="\t", header=None, dtype=str)[1:].reset_index(drop=True)
    return data

train_data = read_file(dataset_path, "train2id.txt")   

class Custom_TrainLoader(PyTorchTrainDataLoader.PyTorchTrainDataLoader):
    def _PyTorchTrainDataLoader__construct_dataset(self, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel):
            f = open(self.ent_file, "r")
            ent_total = (int)(f.readline())
            f.close()

            f = open(self.rel_file, "r")
            rel_total = (int)(f.readline())
            f.close()
            
            data = pd.read_csv(self.tri_file, sep="\t", header=None)

            triples_total = (int)(data.loc[0,0])
            head = data.loc[1:, :][0].astype('int32').tolist()
            tail = data.loc[1:, :][1].astype('int32').tolist()
            rel = data.loc[1:, :][2].astype('int32').tolist()

            dataset = Custom_Dataset(np.array(head), np.array(tail), np.array(rel), ent_total, rel_total, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel)
            return dataset        

class Custom_Dataset(PyTorchTrainDataLoader.PyTorchTrainDataset):
    def _PyTorchTrainDataset__count_htr(self):
        self.h_of_tr = {}
        self.t_of_hr = {}
        self.r_of_ht = {}
        self.h_of_r = {}
        self.t_of_r = {}
        self.freqRel = {}
        self.lef_mean = {}
        self.rig_mean = {}

        for h, t, r in zip(self.head, self.tail, self.rel):

            # initialize distribution of combinations and data structures
            if (h, r) not in self.t_of_hr:
                self.t_of_hr[(h, r)] = set()
            
            if (t, r) not in self.h_of_tr:
                self.h_of_tr[(t, r)] = set()
            
            if (h, t) not in self.r_of_ht:
                self.r_of_ht[(h, t)] = set()
            
            if r not in self.freqRel:
                self.freqRel[r] = 0
                self.h_of_r[r] = set()
                self.t_of_r[r] = set()

            # append head, relation, tail to respective set
            self.t_of_hr[(h, r)].add(t)            
            self.h_of_tr[(t, r)].add(h)
            self.r_of_ht[(h, t)].add(r)

            # counter for unique head & tail & relation combinations
            self.freqRel[r] += 1.0
            self.h_of_r[r].add(h)
            self.t_of_r[r].add(t)

        for t, r in self.h_of_tr:
            self.h_of_tr[(t, r)] = np.array(list(self.h_of_tr[(t, r)]))
        for h, r in self.t_of_hr:
            self.t_of_hr[(h, r)] = np.array(list(self.t_of_hr[(h, r)]))
        for h, t in self.r_of_ht:
            self.r_of_ht[(h, t)] = np.array(list(self.r_of_ht[(h, t)]))
        
        for r in self.freqRel:
            self.h_of_r[r] = np.array(list(self.h_of_r[r]))
            self.t_of_r[r] = np.array(list(self.t_of_r[r]))
            self.lef_mean[r] = self.freqRel[r] / len(self.h_of_r[r])
            self.rig_mean[r] = self.freqRel[r] / len(self.t_of_r[r])

class Custom_Model(TransE):
    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h ,t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

train_dataloader = Custom_TrainLoader(
    in_path=dataset_path+"/",
    nbatches=args["nbatches"],  # number of batches
    threads=8,
    sampling_mode="normal",
    bern_flag=1,  # noise for negative triples
    filter_flag=1,
    neg_ent=25,
    )

print("Loading Model...")

# define the model
transe = Custom_Model(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200,  # embedding dim
    p_norm=1,
    norm_flag=True)

# dataloader for test
#test_dataloader = TestDataLoader(dataset_path)

print("Loading Loss Function...")

# define the loss function
model = NegativeSampling(
    model=transe,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)

print("Training...")

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader,
                  train_times=args["nepochs"], alpha=args["lr"], use_gpu=True)
trainer.run()

print("Saving Checkpoint...")

Path('./checkpoint/').mkdir(parents=True, exist_ok=True)
transe.save_checkpoint('./checkpoint/transe.ckpt')
task.upload_artifact(name='transe.ckpt', artifact_object='./checkpoint/transe.ckpt')

