# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# Robustness to Image Translation
# Demonstrations of (1) Setting up PyTorch training and testin experiments with
# hydra-zen and rai-toolbox (2) Configuring and running repeatable and scalable
# experiments with hydra-zen and (3) Experimenting with robustness to image
# translations on MNIST

#%% Import Modules

# Import standard modules
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
#Bad installs for torch
#plt.figure()
#plt.close('all')

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import xarray as xr
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

# Import from rAI Toolbox
from hydra_zen import MISSING, builds, instantiate, launch, make_config
from rai_toolbox.mushin.workflows import RobustnessCurve

#%% Define if you are using CPU or GPU
device = 'cpu'
#device = 'cuda'

#%% Define Training and Testing Functions and Models

# Set random seed
def set_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
# Experiment functions
class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,5, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32,32,3,padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32,32,3,padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,10,3),
            nn.Flatten(1),            
        )
        
    def forward(self, x):
        return self.model(x)
    
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )
        
    def forward(self,x):
        return self.model(x)
        
# Define translation pertubation function
def translate_pertubation(data:Tensor,epsilon) -> Tensor:
    xform = transforms.RandomAffine(degrees=0.0, translate=(epsilon,epsilon))
    return xform(data)
        
# Define training and testing models
def train_model(
    model,
    train_dataset,
    epsilon=0.0,
    num_epochs=10,
    batch_size=100,
    num_workers=4,
    device='cpu',
    learning_rate=0.1,
):
    
    dl = DataLoader(
        train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,
    )
    
    nn_model = model['nn']
    nn_model = nn_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)
    for i in tqdm(range(num_epochs)):
        nn_model.train()
        for data,target in dl:
            data = data.to(device)
            if epsilon > 0:
                data = translate_pertubation(data,epsilon)
                
            target = target.to(device)
            logit = nn_model(data)
            loss = criterion(logit,target)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
    torch.save(nn_model.state_dict(),'model.ckpt')
            
def test_model(
    model,test_dataset,epsilon=0.0,batch_size=100,num_workers=4,device='cpu',
):
    assert model['ckpt'] is not None
    nn_model = model['nn']
    nn_model.load_state_dict(torch.load(str(model['ckpt'])))
    nn_model = nn_model.to(device)
    
    test_dl = DataLoader(
            test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,
    )
        
    preds = []
    with torch.no_grad():
        nn_model.eval()
        for tdata, ttarget in test_dl:
            tdata = tdata.to(device)
            ttarget = ttarget.to(device)
            
            if epsilon > 0:
                tdata = translate_pertubation(tdata,epsilon)
                
            tlogit = nn_model(tdata)
            
            #Save if predication is correct or not for each samples
            preds.extend((tlogit.argmax(1) == ttarget).float().cpu().numpy())
            
    acc = sum(preds)/len(preds)
    return dict(Accuracy=acc)

#Define plotting functions
def bar_chart(cnn_tests,fc_tests,aug_cnn_tests=None,aug_fc_tests=None):
    fix,ax = plt.subplots()
    
    cnn_no_translate = cnn_tests[0].return_value['Accuracy']
    fc_no_translate = fc_tests[0].return_value['Accuracy']
    
    cnn_translate = cnn_tests[1].return_value['Accuracy']
    fc_translate = fc_tests[1].return_value['Accuracy']
    
    if aug_cnn_tests is None:
        ax.bar(
            [0-0.25/2,1-0.25/2],
            [fc_no_translate*100,fc_translate*100],
            width=0.25,
            label='FC'
            )
        ax.bar(
            [0+0.25/2,1+0.25/2],
            [cnn_no_translate*100,cnn_translate*100],
            width=0.25,
            label='CNN'
            )
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Original Test Data','Translated Test Data'])
        
    else:
        aug_cnn_no_translate = aug_cnn_tests[0].return_value['Accuracy']
        aug_fc_no_translate = aug_fc_tests[0].return_value['Accuracy']
        
        aug_cnn_translate = aug_cnn_tests[1].return_value['Accuracy']
        aug_fc_translate = aug_fc_tests[1].return_value['Accuracy']
        
        ax.bar(
            [0-0.25/2,1-0.25/2,2-0.25/2],
            [fc_no_translate*100,fc_translate*100,aug_fc_translate*100],
            width=0.25,
            label='FC'
            )
        ax.bar(
            [0+0.25/2,1+0.25/2,2+0.25/2],
            [cnn_no_translate*100,cnn_translate*100,aug_cnn_translate*100],
            width=0.25,
            label='CNN'
            )
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(
            ['Original Test Data',
             'Translated Test Data',
             'Translated Test Data /n Augmentations'],
            rotation=30
            )
        
    ax.set_ylabel('% Accuracy on Test Set')
    ax.legend()
    
#%%Using hyra-zen to Build Models

DATA_DIR = Path.home() / 'torch.'/'data'      
ToTensor = builds(transforms.ToTensor)
Dataset = builds(datasets.MNIST,root=DATA_DIR,train=True,transform=ToTensor)
TestDataset = builds(datasets.MNIST,root=DATA_DIR,train=False,transform=ToTensor)

#Test
#Download MNIST dataset: datasets.mnist.MNIST(root=DATA_DIR,download=True)
dataset = instantiate(Dataset)
X,Y = dataset[0]
Xt = translate_pertubation(X,0.3)

fig,ax = plt.subplots(ncols=2,subplot_kw=dict(xticks=[],yticks=[]))
ax[0].imshow(X[0],cmap='Greys')
ax[1].imshow(Xt[0],cmap='Greys')

#Define models
_ConvModelCfg = builds(ConvModel)
_LinearModelCfg = builds(LinearModel)

ConvModelCfg = make_config(ckpt=None,nn=_ConvModelCfg)
LinearModelCfg = make_config(ckpt=None,nn=_LinearModelCfg)

#Test
dataset = instantiate(Dataset)
cnn = instantiate(ConvModelCfg)
fc = instantiate(LinearModelCfg)

X,Y = dataset[0]
assert cnn.nn(X[None]).shape == (1,10)
assert fc.nn(X[None]).shape == (1,10)
        
#Training and testing functions
Trainer = builds(
    train_model,
    populate_full_signature=True,
    train_dataset=Dataset,
    epsilon="${epsilon}",
    zen_partial=True,
)

Tester = builds(
    test_model,
    test_dataset=TestDataset,
    epsilon="${epsilon}",
    populate_full_signature=True,
    zen_partial=True,
)

#Make swappable configurations
cs = ConfigStore.instance()
cs.store(name='cnn',group='model',node=ConvModelCfg)
cs.store(name='fc',group='model',node=LinearModelCfg)

#Main Config        
Config = make_config(
    defaults=['_self_', {'model': 'fc'}],
    epsilon=0.0,
    model=MISSING,
    trainer=Trainer,
    tester=Tester,
)

# store the main config in the ConfigStore
cs.store(name="notebook_config", node=Config)

#%% Train Models
def train_task_fn(cfg):
    # important to set seed BEFORE instantiating any objects
    set_seed(42)

    # recursively instantiates all configurations
    obj = instantiate(cfg)
    return obj.trainer(obj.model)

(training_jobs,) = launch(
    Config,
    train_task_fn,
    overrides=["model=cnn,fc"],
    multirun=True,
)

#Store model results in checkpoint file
TrainedConvModelCfg = make_config(
    ckpt=Path(training_jobs[0].working_dir).absolute() / "model.ckpt",
    nn=_ConvModelCfg,
    hydra_convert="all",
)
TrainedLinearModelCfg = make_config(
    ckpt=Path(training_jobs[1].working_dir).absolute() / "model.ckpt",
    nn=_LinearModelCfg,
    hydra_convert="all",
)

cs.store(name="trained_cnn", group="model", node=TrainedConvModelCfg)
cs.store(name="trained_fc", group="model", node=TrainedLinearModelCfg)      
        
#%% Test the Trained Models

#Define helper function
def testing_task_fn(cfg):
    set_seed(42)
    obj = instantiate(cfg)
    return obj.tester(obj.model)

#Test
(cnn_tests,) = launch(
    Config,
    testing_task_fn,
    overrides=["model=trained_cnn", "epsilon=0,0.2"],
    multirun=True,
)

(fc_tests,) = launch(
    Config,
    testing_task_fn,
    overrides=["model=trained_fc", "epsilon=0,0.2"],
    multirun=True,
)

#Plot results
bar_chart(cnn_tests, fc_tests)
        
#%% Introduce the Mushin Robustness Curve Class to Help Evaluate

class TranslationRobustness(RobustnessCurve):
    @staticmethod
    def task(model, tester) -> torch.Tensor:
        set_seed(42)
        return tester(model)

#Run the class
task = TranslationRobustness(Config)
task.run(epsilon="range(0, 1, 0.1)", model="trained_cnn")

#Plot results
fig, ax = plt.subplots()
#task.plot("Accuracy", ax=ax)

#Example using mutli-run option for mutliple models
task = TranslationRobustness(Config)
task.run(epsilon='range(0,1,0.1)',model='trained_fc,trained_cnn')

fig,ax = plt.subplots()
#task.plot('Accuracy',ax=ax,group='model')

#Convert to xarray format and plot results
xdata = task.to_xarray()
fig, ax = plt.subplots()
for name, g in xdata.groupby("model"):
    for k in g.data_vars:
        g[k].plot.line(x="epsilon", ax=ax, label=name)
plt.legend()

# Take-Aways:
#
# Domain Knowledge: CNNs model translation equivariance into the model, an inductive bias based on domain knowledge.
# Robustness: CNNs are more robust to unseen translations than a linear network.
# Complexity: CNNs are less complex (less parameters) than linear networks.
# Sample Efficient: CNNs acheive top performance metrics with less data.


#%% Data Augmentations
#What if we train the models on the 'corrupted' data?

#Train models on epsilon=0.2 data
(aug_training_jobs,) = launch(
    Config,
    train_task_fn,
    overrides=["model=cnn,fc", "trainer.device="+device, "epsilon=0.2"],
    multirun=True,
)

#Create new config files with checkpoints
AugTrainedConvModelCfg = make_config(
    ckpt=Path(aug_training_jobs[0].working_dir).absolute() / "model.ckpt",
    nn=_ConvModelCfg,
    hydra_convert="all",
)
AugTrainedLinearModelCfg = make_config(
    ckpt=Path(aug_training_jobs[1].working_dir).absolute() / "model.ckpt",
    nn=_LinearModelCfg,
    hydra_convert="all",
)

cs.store(name="aug_trained_cnn", group="model", node=AugTrainedConvModelCfg)
cs.store(name="aug_trained_fc", group="model", node=AugTrainedLinearModelCfg)

#Compare the bar charts of the tested data
(aug_cnn_tests,) = launch(
    Config,
    testing_task_fn,
    overrides=["model=aug_trained_cnn", "epsilon=0,0.2",],
    multirun=True,
)

(aug_fc_tests,) = launch(
    Config,
    testing_task_fn,
    overrides=["model=aug_trained_fc", "epsilon=0,0.2",],
    multirun=True,
)

bar_chart(cnn_tests, fc_tests, aug_cnn_tests, aug_fc_tests)

#Generate robustness curves for comparison
task_aug = TranslationRobustness(Config)
task_aug.run(epsilon="range(0,1,0.1)", model="aug_trained_fc,aug_trained_cnn")

fig, ax = plt.subplots()
task_aug.plot("Accuracy", ax=ax, group="model")

#Convert to xarray data and compile with original model
xdata = task.to_xarray()
aug_xdata = task_aug.to_xarray()
all_xdata = xr.concat((xdata, aug_xdata), dim="model")
all_xdata

#Plot comparison of results
fig, ax = plt.subplots()
for name, g in all_xdata.groupby("model"):
    for k in g.data_vars:
        g[k].plot.line(x="epsilon", ax=ax, label=name)
plt.legend()
















