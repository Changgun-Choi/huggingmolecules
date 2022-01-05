# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:47:01 2022

@author: ChangGun Choi

"""
#!pip install repackage
#!pip install -r huggingmolecules/experiments/requirements.txt
#!pip install -c conda-forge rdkit==2020.09.1
!pip install --upgrade gin-config
!pip install --upgrade -q gin git+https://github.com/google/trax.git@v1.2.3
#!pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install -e "C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules/src"
!pip install jaxlib
!pip install jax[cuda111] -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

import sys
import os
os.chdir('C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules')


from huggingmolecules import MatModel, MatFeaturizer
from experiments.src import TrainingModule, get_data_loaders

model = MatModel.from_pretrained('mat_masking_20M')
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')
