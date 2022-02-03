# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:47:01 2022

@author: ChangGun Choi

"""
#!pip install repackage
!pip install -r "C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/experiments/requirements.txt"
#!pip install -c conda-forge rdkit==2020.09.1
!pip install --upgrade gin-config
!pip install --upgrade -q gin git+https://github.com/google/trax.git@v1.2.3
#!pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install -e "C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/src"
!pip install jaxlib
!pip install jax[cuda111] -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
!pip install git+https://https://github.com/google/gin-config
#%%

import sys
import os
os.chdir('C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules')

import gin
#!pip install --upgrade gin-config
import random
#import gin.torch
from huggingmolecules import MatModel, MatFeaturizer

# The following import works only from the source code directory:
#!pip install gin-config==0.1.4
from experiments.src import TrainingModule, get_data_loaders
from experiments.src import *
#from torch.nn import MAELoss
from torch.nn import L1Loss
from torch.optim import Adam

from pytorch_lightning import Trainer
from pytorch_lightning.metrics import MeanAbsoluteError
import pandas as pd
#%%
from experiments.src.training import *

#%%
model_flash = MatModel.from_pretrained('mat_masking_20M')
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')

#cd "C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/data"
train_data = pd.read_csv("C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/data/flash_train.csv")
train_data
 

_get_data_split_from_csv(dataset_name: str,
                             assay_name: str,
                             dataset_path: "C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/data/flash_train.csv",
                             split_method: str,
                             split_frac: Tuple[float, float, float],
                             split_seed: int)

get_data_split(task_name: str,
                   dataset_name: str,
                   assay_name: str = None,
                   split_method: str = "random",
                   split_frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   split_seed: Union[int, str] = None,
                   normalize_labels: bool = False,
                   dataset_path: str = None) 


get_data_loaders(featurizer: PretrainedFeaturizerMixin, *,
                    batch_size: int,
                    num_workers: int = 0,
                    cache_encodings: bool = False,
                    task_name: str = None,
                    dataset_name: str = None)


#%%
batch_size = 32
num_workers = 0

train_loader = featurizer.get_data_loader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)


valid_loader = featurizer.get_data_loader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = featurizer.get_data_loader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#%%
# Build the pytorch lightning training module:
pl_module = TrainingModule(model,
                           loss_fn=L1Loss(),
                           metric_cls=MeanAbsoluteError,
                           optimizer=Adam(model.parameters()))

# Build the data loader for the freesolv dataset:
train_dataloader, _, _ = get_data_loaders(featurizer,
                                          batch_size=32,
                                          task_name=None,
                                          dataset_name=None)

# Build the pytorch lightning trainer and fine-tune the module on the train dataset:
trainer = Trainer(max_epochs=1)
trainer.fit(pl_module, train_dataloader=train_dataloader)

# Make the prediction for the batch of SMILES strings:
batch = featurizer(['C/C=C/C', '[C]=O'])
output = pl_module.model(batch)
output

#sep='/t' 
#tdc.utils.retrieve_label_name_list('tox21')
#get_data_split  : huggingmolecules/experiments/src/training/training_utils.py

#%%
# Save the weights of the model (usually after the fine-tuning process):
model.save_weights('tuned_mat_masking_20M.pt')

# Load the previously saved weights
# (which now includes all layers of the model):
model.load_weights('tuned_mat_masking_20M.pt')

# Load the previously saved weights, but without 
# the last layer of the model ('generator' in the case of the 'MatModel')
model.load_weights('tuned_mat_masking_20M.pt', excluded=['generator'])

# Build the model and load the previously saved weights:
config = MatConfig.from_pretrained('mat_masking_20M')
model = MatModel.from_pretrained('tuned_mat_masking_20M.pt',
                                 excluded=['generator'],
                                 config=config)





#%%
from huggingmolecules import MatConfig, MatFeaturizer, MatModel

# Build the model with the pre-defined config:
config = MatConfig.from_pretrained('mat_masking_20M')
model = MatModel(config)

# Load the pre-trained weights 
# (which do not include the last layer of the model)
model.load_weights('mat_masking_20M')

# Build the model and load the pre-trained weights in one line:
model = MatModel.from_pretrained('mat_masking_20M')

# Encode (featurize) the batch of two SMILES strings: 
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')
batch = featurizer(['C/C=C/C', '[C]=O'])
batch
# Feed the model with the encoded batch:
output = model(batch)
output
# Save the weights of the model (usually after the fine-tuning process):
model.save_weights('tuned_mat_masking_20M.pt')

# Load the previously saved weights
# (which now includes all layers of the model):
model.load_weights('tuned_mat_masking_20M.pt')

# Load the previously saved weights, but without 
# the last layer of the model ('generator' in the case of the 'MatModel')
model.load_weights('tuned_mat_masking_20M.pt', excluded=['generator'])

# Build the model and load the previously saved weights:
config = MatConfig.from_pretrained('mat_masking_20M')
model = MatModel.from_pretrained('tuned_mat_masking_20M.pt',
                                 excluded=['generator'],
                                 config=config)



#%%
import numpy as np
cm = [[ 321,   26,    6,    2],
 [  36,  675,   48,   13],
 [   1,   58,  804,  142],
 [   0,    7,   57, 1663]]
cm = np.array(cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
a = cm.sum(axis=1)
b = cm.sum(axis=1)[:,np.newaxis]
cm












cm.astype('float')



