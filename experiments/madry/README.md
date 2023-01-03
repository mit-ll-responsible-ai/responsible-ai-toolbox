# Reproduce Results From Madry Lab's Robustness Toolbox

This experiment takes pre-trained robust models developed at [MadryLab](https://github.com/MadryLab/robustness) and reproduces results for CIFAR-10 and [Restricted ImageNet](https://github.com/MadryLab/robust_representations) datasets. For CIFAR-10, we aim to reproduce the results for the $\ell_2$-norm shown in this [README](https://github.com/MadryLab/robustness/blob/master/README.rst). For Restricted ImageNet we want to reproduce the results shown in shown in Appendix A.4 from https://arxiv.org/abs/1906.00945.

The pretrained models for this experiment can be found here(WILL ADD LINK LATER):

  - `mitll_cifar_l2_1_0`: A simplified data structure of the pre-trained model available via the [README](https://github.com/MadryLab/robustness/blob/master/README.rst). This is a ResNet-50 model trained with Projected Gradient Descent (PGD) using the $\ell_2$-norm with $\epsilon=1.0$
  - `mitll_restricted_imagenet_l2_3_0`: TA simplified data structure of the pre-trained model available via the [README](https://github.com/MadryLab/robust_representations/blob/master/README.md).  This is a ResNet-50 model trained with PGD using the $\ell_2$-norm with $\epsilon=3.0$.


# Prerequisites
  - hydra-zen
  - torchvision
  - PyTorch Lightning
  - torchmetrics

## Exeperiments


### CIFAR10
Reproduce last column of "CIFAR10 L2-robust accuracy" table from https://github.com/MadryLab/robustness by executing PGD with multiple $\epsilon$ values and the robustly trained ResNet-50 model.  These results are for 20-step PGD with step size of `2.5 * ε-test / num_steps`.

```
PYTHONPATH=$PWD:$PYTHONPATH python run.py ckpt=mitll_cifar_l2_1_0.pt steps=20 epsilon=[0,0.25,0.5,1.0,2.0]
```

or in an interactive python environment such as a notebook:

```python
import os

import configs
import hydra
import pytorch_lightning as pl
from rai_toolbox.mushin.workflows import RobustnessCurve

# Need add current working directory to the python path
os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.getenv('PYTHONPATH')}"

# Implement the evaluation task for RobustnessCurv
class MadryLabRobustness(RobustnessCurve):
    @staticmethod
    def task(
        seed: int, trainer: pl.Trainer, module: pl.LightningModule
    ) -> dict:
        pl.seed_everything(seed)
        trainer.test(module)
        assert Path("test_metrics.pt").exists()
        return tr.load("test_metrics.pt")

robustness_job = MadryLabRobustness(configs.Config)
robustness_job.run(
    epsilon=[0, 0.25, 0.5, 1.0, 2.0],
    ckpt="mitll_cifar_l2_1_0.pt"
)
robustness_job.plot("Test/Accuracy")
```

#### Results

|Test-eps | Expected | rai_toolbox    |
|---------|----------|----------------|
| 0.0     | 82       | 82             |
| 0.25    | 76       | 76             |
| 0.5     | 69       | 69             |
| 1.0     | 53       | 53             |
| 2.0     | 15       | 19             |

Small difference for $\epsilon=2.0$.

#### Training a Standard and Robust Model

To train a standard CIFAR-10 model:

```
python run.py mushin/dataset=cifar10 mushin/model=cifar10_resnet50 +mushin/optim=cifar ~mushin/perturbation
```

To train a robust CIFAR-10 model:

```
python run.py mushin/dataset=cifar10 mushin/model=cifar10_resnet50 +mushin/optim=cifar mushin.perturbation.epsilon=1.0 mushin.perturbation.steps=7
```


### Restricted ImageNet

Reproduce test results shown in Appendix A.4 from https://arxiv.org/abs/1906.00945 by executing PGD with multiple $\epsilon$ values and the robustly trained ResNet-50 model. These results are for 20-step PGD with step size of `2.5 * ε-test / num_steps`.

```
python run.py data_path=$HOME/.torch/data/imagenet/normal/val ckpt=mitll_restricted_imagenet_l2_3_0.pt mushin/dataset=restricted_imagenet mushin/model=resnet50 mushin.perturbation.steps=20 mushin.perturbation.epsilon=3 
```

#### Results

|Test-eps | Expected | rai_toolbox    |
|---------|----------|----------------|
| 3.0     | 82       | 82             |



## 📄 Citations

```
@misc{robustness,
   title={Robustness (Python Library)},
   author={Logan Engstrom and Andrew Ilyas and Hadi Salman and Shibani Santurkar and Dimitris Tsipras},
   year={2019},
   url={https://github.com/MadryLab/robustness}
}

@article{engstrom2019adversarial,
  title={Adversarial robustness as a prior for learned representations},
  author={Engstrom, Logan and Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Tran, Brandon and Madry, Aleksander},
  journal={arXiv preprint arXiv:1906.00945},
  year={2019}
}
```

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

© 2023 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
