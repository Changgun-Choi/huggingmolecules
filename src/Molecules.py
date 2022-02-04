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
!pip install rdkit-pypi
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
#from help import *
import logging

#%%
from experiments.src.training import *
from rdkit.Chem.rdmolfiles import MolFromSmiles

#%%

#cd "C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/data"

#train_smiles = train_data['smiles'].apply(lambda x: str(x))

def preprocess(data):
    
    smiles = data['smiles']        
    drop = []
    for i, smile in enumerate(smiles):
        mol = MolFromSmiles(smile)
        
        if mol == None:
            drop.append(i)
            
        #AllChem.UFFOptimizeMolecule(mol)
    print(drop)
    
    data = data.drop(drop)
    dataset = data.reset_index(drop=True)
    
    return dataset
#%%
train_data = pd.read_csv("C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/data/flash_train.csv")
valid_data = pd.read_csv("C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/data/flash_valid.csv")
test_data = pd.read_csv("C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/data/flash_test.csv")

train_data = preprocess(train_data)
train_data
#preprocess(valid_data)
#preprocess(test_data)


#%%

def split_data(train_data,valid_data, test_data):
    
    return {
        'train': {
                  'X': train_data['smiles'].to_list(),
                  'Y': train_data['y'].to_numpy()},
        'valid': {
                  'X': valid_data['smiles'].to_list(),
                  'Y': valid_data['y'].to_numpy()},
        'test': {
                 'X': test_data['smiles'].to_list(),
                 'Y': test_data['y'].to_numpy()}
    }


split = split_data(train_data,valid_data, test_data)


#%%
from typing import Tuple, List, Union, Type, Callable, Dict, Any
from torch.utils.data import DataLoader, random_split
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin


def data_loader(featurizer=PretrainedFeaturizerMixin,*, batch_size, num_workers,cache_encodings: bool = False, split) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    split['train']['X'] = featurizer.encode_smiles_list(split['train']['X'], split['train']['Y'])
    split['valid']['X'] = featurizer.encode_smiles_list(split['valid']['X'], split['valid']['Y'])
    split['test']['X'] = featurizer.encode_smiles_list(split['test']['X'], split['test']['Y'])
    
    if cache_encodings and not _encodings_cached():
        _dump_encodings_to_cache(split)
    
    train_data = split['train']['X']
    valid_data = split['valid']['X']
    test_data = split['test']['X']

    #logging.info(f'Train samples: {len(train_data)}')
    #logging.info(f'Validation samples: {len(valid_data)}')
    #logging.info(f'Test samples: {len(test_data)}')

    train_loader = featurizer.get_data_loader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = featurizer.get_data_loader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = featurizer.get_data_loader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

#%%


batch_size = 32
num_workers = 0
model = MatModel.from_pretrained('mat_masking_20M')
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')

train_dataloader, _, _ = data_loader(featurizer, batch_size=32 ,num_workers=0, split=split)


#  File "C:\Users\CHANGG~1\AppData\Local\Temp/ipykernel_16548/1257150278.py", line 1, in <listcomp>
 #   np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
#  NameError: name 'get_atom_features' is not defined

#%%
# Build the pytorch lightning training module:
pl_module = TrainingModule(model,
                           loss_fn=L1Loss(),
                           metric_cls=MeanAbsoluteError,
                           optimizer=Adam(model_flash.parameters()))

# Build the pytorch lightning trainer and fine-tune the module on the train dataset:
trainer = Trainer(max_epochs=5)
trainer.fit(pl_module, train_dataloader=train_dataloader)
#%%
# Make the prediction for the batch of SMILES strings:
#batch = featurizer('CCCCCCCCCCCO')    
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



