#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:11:12 2022

@author: fixitfelix
"""

from clearml import Task, StorageManager, Dataset

import os
import pandas as pd
import numpy as np
import tqdm
import multiprocessing

from openke.config import Trainer, Tester
from openke.module.model import TransE
import ipdb
import torch

remote_path = "s3://experiment-logging"
#task = Task.init(project_name='gdelt-embeddings', task_name='graph-clustering',
#                 output_uri=os.path.join(remote_path, "storage"))
#task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
#task.execute_remotely(queue_name="compute2", exit_process=True)


dataset_obj = Dataset.get(dataset_project="datasets/gdelt", dataset_name="gdelt_openke_format_w_extras", only_published=True)
dataset_path = dataset_obj.get_local_copy()
print(list(os.walk(dataset_path)))

def read_file(dataset_path:str, file_name:str, sep="\t"):
    data = pd.read_csv('{}/{}'.format(dataset_path, file_name), sep=sep, header=None, dtype=str)[1:].reset_index(drop=True)
    return data

entity_dict = read_file(dataset_path, "entity2id.txt")
relation_dict = read_file(dataset_path, "relation2id.txt")
train_data = read_file(dataset_path, "train2id.txt")
document = read_file(dataset_path, "url2id.txt", sep=",")


document_filtered = document[(document[1].str.contains("")) & (document[1].str.contains("south-china-sea"))]
train_data[5] = train_data[4].str[:6].astype("int32")
train_data=train_data[train_data[5]<202101]
results = document_filtered.merge(train_data, how="inner", left_on=0, right_on=3)

import json, os
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def process_text(url:str):
    return url.strip("/").split("/")[-1].replace("-", " ")

cluster_id = "153"
cluster_json_path = "/mnt/projects/AI_Projects/gdelt_retrieval/data/cluster_data.json"

def get_corpus_from_cluster(cluster_id:str, cluster_json_path:str)->pd.DataFrame:
    clusters = json.load(open(cluster_json_path))
    cluster_ids_size = {cluster_id: len(clusters[cluster_id]["id_list"]) for cluster_id in clusters.keys()}
    largest_cluster = pd.DataFrame(clusters[cluster_id]["id_list"])
    largest_cluster_url = largest_cluster.merge(document, how="left", left_on=0, right_on=0)
    largest_cluster_url[2]=largest_cluster_url[1].apply(lambda x: process_text(x))
    return largest_cluster_url, cluster_ids_size

def plot_wordcloud(corpus:str, cluster_id:str)->None:    
    #Path("/home/fixitfelix/Desktop/word_cloud").mkdir(parents=True, exist_ok=True)
    word_cloud = WordCloud(collocations=False, background_color="white").generate(corpus)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("/home/fixitfelix/Desktop/word_cloud/{}_wordcloud.png".format(cluster_id))
    plt.close()
    

def run(cluster_id, cluster_json_path):
    largest_cluster_url, cluster_ids_size = get_corpus_from_cluster(cluster_id, cluster_json_path)
    corpus = " ".join(largest_cluster_url[2].tolist())
    plot_wordcloud(corpus, cluster_id)

run("191", cluster_json_path)


#from sklearn.decomposition import LatentDirichletAllocation as lda
#from sklearn.datasets import make_multilabel_classification

#x,_=make_multilabel_classification(random_state=0)

#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1, stop_words="english")
#count_vectorizer.fit_transform(largest_cluster_url[2])
#vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1, stop_words="english")
#results = vectorizer.fit_transform(largest_cluster_url[2])
#cluster_model = lda(n_components=1)
#cluster_results = cluster_model.fit_transform(results)





#train_data[3]==



