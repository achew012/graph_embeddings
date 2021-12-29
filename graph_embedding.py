from clearml import Task, StorageManager, Dataset
import argparse
import json

#Task.add_requirements('transformers', package_version='4.2.0')
#task = Task.init(project_name='gdelt-embeddings', task_name='training', output_uri="s3://experiment-logging/storage/")
#task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")

#docker_setup_bash_script=['git clone https://github.com/thunlp/OpenKE', 'cd OpenKE/openke', 'bash make.sh']
#task.connect(args)
#task.execute_remotely(queue_name="compute2", exit_process=True)
#clearlogger = task.get_logger()

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./data/openke_format/", 
	nbatches = 1,
	threads = 4, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
#test_dataloader = TestDataLoader("./data/openke_format/")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 10, 
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 500, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

