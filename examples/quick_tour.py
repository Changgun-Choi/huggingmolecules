import os
import sys
import os
os.chdir('C:/Users/ChangGun Choi/Team Project/Molecules/huggingmolecules')
#%%

from huggingmolecules import MatModel, MatFeaturizer

sys.path.insert(0, os.path.abspath('..'))


# The following import works only from the source code directory:
from experiments.src import TrainingModule, get_data_loaders

from torch.nn import MSELoss
from torch.optim import Adam

from pytorch_lightning import Trainer
from pytorch_lightning.metrics import MeanSquaredError

# Build and load the pre-trained model and the appropriate featurizer:
model = MatModel.from_pretrained('mat_masking_20M')
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')

# Build the pytorch lightning training module:
pl_module = TrainingModule(model,
                           loss_fn=MSELoss(),
                           metric_cls=MeanSquaredError,
                           optimizer=Adam(model.parameters()))

# Build the data loader for the freesolv dataset:
train_dataloader, _, _ = get_data_loaders(featurizer,
                                          batch_size=32,
                                          task_name='ADME',
                                          dataset_name='hydrationfreeenergy_freesolv')
#%%
# Build the pytorch lightning trainer and fine-tune the module on the train dataset:
trainer = Trainer(max_epochs=7)
trainer.fit(pl_module, train_dataloader=train_dataloader)

#%%
# Make the prediction for the batch of SMILES strings:
featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')

batch1 = featurizer(['C/C=C/C'])
batch1
batch2 = featurizer(['[C]=O'])
batch3 = featurizer(['CCCCCCCCCCCO'])
batch3
print(pl_module.model(batch1))
print(pl_module.model(batch2))
print(pl_module.model(batch3))