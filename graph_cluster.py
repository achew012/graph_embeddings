from clearml import Task, StorageManager, Dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", help="increase output verbosity")
args = parser.parse_args()

import os
import pandas as pd
import numpy as np
import tqdm
import multiprocessing

from openke.config import Trainer, Tester
from openke.module.model import TransE
import torch
#import ipdb


remote_path = "s3://experiment-logging"
task = Task.init(project_name='gdelt-embeddings', task_name='graph-document-embeddings-{}'.format(args.date),
                 output_uri=os.path.join(remote_path, "storage"))
task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
task.execute_remotely(queue_name="compute2", exit_process=True)

dataset_obj = Dataset.get(dataset_project="datasets/gdelt", dataset_name="gdelt_openke_format_w_extras", only_published=True)
dataset_path = dataset_obj.get_local_copy()
print(list(os.walk(dataset_path)))

def read_file(dataset_path:str, file_name:str):
    data = pd.read_csv('{}/{}'.format(dataset_path, file_name), sep="\t", header=None, dtype=str)[1:].reset_index(drop=True)
    return data

entity_dict = read_file(dataset_path, "entity2id.txt")
relation_dict = read_file(dataset_path, "relation2id.txt")
train_data = read_file(dataset_path, "train2id.txt")
train_data[5] = train_data[4].str[:6]

transe = TransE(
    ent_tot=len(entity_dict),
    rel_tot=len(relation_dict),
    dim=200, # embedding dim
    p_norm=1,
    norm_flag=True
    )

weights_path = 's3://experiment-logging/storage/gdelt-embeddings/openke-graph-training.3ed8b6262cd34d52b092039d0ee1d374/artifacts/transe.ckpt/transe.ckpt'
weights_path = StorageManager.get_local_copy(remote_url=weights_path)
transe.load_checkpoint(weights_path)

print(transe.ent_embeddings.weight.size())
print(transe.rel_embeddings.weight.size())
#print(train_data)

# def collate_entity_embeddings_from_model(ent_list:list):
#     embedding = list(map(lambda x:transe.ent_embeddings.weight[x], ent_list))
#     doc_embedding = np.sum(embedding) / len(embedding)
#     return doc_embedding

def collate_entity_embeddings_from_model(triples_df: pd.DataFrame):
    src_entities = torch.tensor(triples_df[0].astype("int32").tolist())
    tgt_entities = torch.tensor(triples_df[1].astype("int32").tolist())
    relations = torch.tensor(triples_df[2].astype("int32").tolist())
    src_embeddings = transe.ent_embeddings(src_entities) 
    tgt_embeddings = transe.ent_embeddings(tgt_entities) 
    relations_embeddings = transe.rel_embeddings(relations) 
    doc_embedding = torch.cat([src_embeddings, relations_embeddings, tgt_embeddings], dim=1).mean(dim=0)
    return doc_embedding


def get_doc_embeddings(triples_per_temporal_interval:pd.DataFrame, doc:str):
    triples_in_doc = triples_per_temporal_interval[triples_per_temporal_interval[3]==doc].loc[:,:2]
    doc_embedding = collate_entity_embeddings_from_model(triples_in_doc)

    #unique_entities_in_doc = set(triples_in_doc[0].astype(int))
    #unique_entities_in_doc.update(triples_in_doc[1].astype(int))
    #unique_entities_in_doc = list(unique_entities_in_doc)
    #doc_embedding = collate_entity_embeddings_from_model(unique_entities_in_doc)
    return doc_embedding

def get_document_embeddings_from_interval(unique_urls_per_interval:list, triples_per_temporal_interval:pd.DataFrame):
    doc_embedding_dict = {}
    for idx, doc in enumerate(tqdm.tqdm(unique_urls_per_interval)):
        doc_embedding = get_doc_embeddings(triples_per_temporal_interval, str(doc))
        doc_embedding_dict[doc] = doc_embedding
    return doc_embedding_dict

def process_interval(date):
    triples_per_temporal_interval = train_data[train_data[5]==str(date)].drop(columns=5)
    unique_urls_per_interval = triples_per_temporal_interval[3].unique().tolist()
    doc_embedding_dict = get_document_embeddings_from_interval(unique_urls_per_interval, triples_per_temporal_interval)
    return doc_embedding_dict

# def get_all_embeddings(interval_list:list):
#     return [process_interval(date) for date in interval_list if date<202101]

unique_dates = pd.Series(train_data[5].unique()).astype(int).sort_values(ascending=True)
#temporal_list = get_all_embeddings(unique_dates)
# for date in unique_dates:
#     if date<202101:
#         process_interval(date)

temporal_list = process_interval(args.date)
task.upload_artifact(name='temporal_list_by_idx', artifact_object=temporal_list)

