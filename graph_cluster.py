from clearml import Task, StorageManager, Dataset

import os
import pandas as pd
import numpy as np
import tqdm
import multiprocessing

from dgl.data import dgl_dataset
from openke.config import Trainer, Tester
from openke.module.model import TransE

dataset_obj = Dataset.get(dataset_project="datasets/gdelt", dataset_name="gdelt_openke_format_w_extras", only_published=True)
dataset_path = dataset_obj.get_local_copy()
print(list(os.walk(dataset_path)))

def read_file(dataset_path:str, file_name:str):
    data = pd.read_csv('{}/{}'.format(dataset_path, file_name), sep="\t", header=None, dtype=str)[1:].reset_index(drop=True)
    return data

entity_dict = read_file(dataset_path, "entity2id.txt")
relation_dict = read_file(dataset_path, "relation2id.txt")
train_data = read_file(dataset_path, "train2id.txt")[:250000]

transe = TransE(
    ent_tot=11384,
    rel_tot=253,
    dim=50,# embedding dim
    p_norm=1,
    norm_flag=True
    )

transe.load_checkpoint('./checkpoint/transe.ckpt')

#print(transe.ent_embeddings.weight.size())
#print(transe.rel_embeddings.weight.size())
#print(train_data)

def collate_entity_embeddings_from_model(ent_list:list):
    embedding = list(map(lambda x:transe.ent_embeddings.weight[x], ent_list))
    doc_embedding = np.sum(embedding) / len(embedding)
    return doc_embedding

def get_doc_embeddings(triples_per_temporal_interval:pd.DataFrame, doc:str):
    triples_in_doc = triples_per_temporal_interval[triples_per_temporal_interval[3]==doc]
    unique_entities_in_doc = set(triples_in_doc[0].astype(int))
    unique_entities_in_doc.update(triples_in_doc[1].astype(int))
    unique_entities_in_doc = list(unique_entities_in_doc)
    doc_embedding = collate_entity_embeddings_from_model(unique_entities_in_doc)
    return doc_embedding

def get_document_embeddings_from_interval(unique_urls_per_interval:list, triples_per_temporal_interval:pd.DataFrame):
    doc_embedding_dict = {}
    for idx, doc in enumerate(tqdm.tqdm(unique_urls_per_interval)):
        doc_embedding = get_doc_embeddings(triples_per_temporal_interval, str(doc))
        doc_embedding_dict[doc] = doc_embedding
    return doc_embedding_dict

def process_interval(date):
    #print("Processing period {} of {}".format(idx+1, len(unique_dates)))
    triples_per_temporal_interval = train_data[train_data[5]==str(date)].drop(columns=5)
    unique_urls_per_interval = triples_per_temporal_interval[3].unique().tolist()
    doc_embedding_dict = get_document_embeddings_from_interval(unique_urls_per_interval, triples_per_temporal_interval)
    return doc_embedding_dict

def get_all_embeddings(interval_list:list):
     return [process_interval(date) for date in interval_list]

# =============================================================================
# def get_all_embeddings(interval_list:list):
#     with multiprocessing.Pool(4) as pool:
#         results = pool.map(process_interval, interval_list)
#     return results 
# =============================================================================

train_data[5] = train_data[4].str[:6]
unique_dates = pd.Series(train_data[5].unique()).astype(int).sort_values(ascending=True)
temporal_list = get_all_embeddings(unique_dates)

# =============================================================================
# temporal_dict = {}
# for idx, date in enumerate(unique_dates):
#     print("Processing period {} of {}".format(idx+1, len(unique_dates)))
#     triples_per_temporal_interval = train_data[train_data[5]==str(date)].drop(columns=5)
#     unique_urls_per_interval = triples_per_temporal_interval[3].unique().tolist()
#     print("Collecting embeddings...")
#     doc_embedding_dict = get_document_embeddings_from_interval(unique_urls_per_interval)
#     temporal_dict[date] = doc_embedding_dict
# =============================================================================



