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
!pip install gin-config==0.1.4
#%%

import sys
import os
os.chdir('C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules')

import gin
import random
import gin.torch
from huggingmolecules import MatModel, MatFeaturizer

# The following import works only from the source code directory:
from experiments.src import TrainingModule, get_data_loaders

#from torch.nn import MAELoss
from torch.nn import L1Loss
from torch.optim import Adam

from pytorch_lightning import Trainer
from pytorch_lightning.metrics import MeanAbsoluteError

#%%
model = MatModel.from_pretrained('mat_masking_20M')
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')

#%%
# Build the pytorch lightning training module:
pl_module = TrainingModule(model,
                           loss_fn=L1Loss(),
                           metric_cls=MeanAbsoluteError,
                           optimizer=Adam(model.parameters()))

# Build the data loader for the freesolv dataset:
train_dataloader, _, _ = get_data_loaders(featurizer,
                                          batch_size=32,
                                          task_name='ADME',
                                          dataset_name='hydrationfreeenergy_freesolv')

# Build the pytorch lightning trainer and fine-tune the module on the train dataset:
trainer = Trainer(max_epochs=1)
trainer.fit(pl_module, train_dataloader=train_dataloader)

# Make the prediction for the batch of SMILES strings:
batch = featurizer(['C/C=C/C', '[C]=O'])
output = pl_module.model(batch)













