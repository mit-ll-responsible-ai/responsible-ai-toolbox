# Reproduce Results From Madry Lab's Robustness Toolbox

This experiment takes pre-trained robust models developed at [MadryLab](https://github.com/MadryLab/robustness) and reproduces results for CIFAR-10 and [Restricted ImageNet](https://github.com/MadryLab/robust_representations) datasets. For CIFAR-10, we aim to reproduce the results for the $\ell_2$-norm shown in this [README](https://github.com/MadryLab/robustness/blob/master/README.rst). For Restricted ImageNet we want to reproduce the results shown in shown in Appendix A.4 from https://arxiv.org/abs/1906.00945.

The pretrained models for this experiment can be found here(WILL ADD LINK LATER):

  - `mitll_cifar_l2_1_0`: A simplified data structure of the pre-trained model available via the [README](https://github.com/MadryLab/robustness/blob/master/README.rst). This is a ResNet-50 model trained with Projected Gradient Descent (PGD) using the $\ell_2$-norm with $\epsilon=1.0$
  - `mitll_restricted_imagenet_l2_3_0`: TA simplified data structure of the pre-trained model available via the [README](https://github.com/MadryLab/robust_representations/blob/master/README.md).  This is a ResNet-50 model trained with PGD using the $\ell_2$-norm with $\epsilon=3.0$.


# Prerequisites
  - hydra-zen
  - torchvision
  - torchmetrics

## Exeperiments


### CIFAR10

Reproduce last column of "CIFAR10 L2-robust accuracy" table from https://github.com/MadryLab/robustness by executing PGD with multiple $\epsilon$ values and the robustly trained ResNet-50 model.  These results are for 20-step PGD with step size of `2.5 * ε-test / num_steps`.

Experiment Configurations:
    - standard: standard training (full dataset)
    - robust: adversarial training (full dataset)
    - standard_fast: standard training with only 100 examples
    - robust_fast: adversarial training with only 100 examples

1. CPU (single process)

```bash
$ python madry.py +experiment=standard_fast
standard [0.9570, 0.1063, 0.0216, 0., 0.]

$ python madry.py +experiment=robust_fast
robust [0.8251, 0.7855, 0.6789, 0.4522, 0.1719]
```

2. Torch Distributed (2 GPU)

```bash
$ torchrun --nproc_per_node=2 madry.py +experiment=standard_fast
standard [0.9570, 0.0895, 0.0091, 0., 0.]

$ torchrun --nproc_per_node=2 madry.py +experiment=standard_fast
robust [0.8251, 0.7855, 0.6726, 0.4431, 0.1719]
```

3. Multiple Experiments (2 GPU)

```bash
$ torchrun --nproc_per_node=2 madry.py +experiment=robust,standard --multirun
[2023-05-24 16:27:48,649][HYDRA] Launching 2 jobs locally
[2023-05-24 16:27:48,649][HYDRA]        #0 : +experiment=robust
robust [0.8158999681472778, 0.7568999528884888, 0.6894999742507935, 0.5347999334335327, 0.19189998507499695]

[2023-05-24 16:28:00,911][HYDRA]        #1 : +experiment=standard
standard [0.950700044631958, 0.09849999845027924, 0.002500000176951289, 0.0, 0.0]
```
#### Results

Here are the results from the Madry Lab's Robustness Toolbox and the results from this experiment.

|Test-eps | Expected | rai_toolbox    |
|---------|----------|----------------|
| 0.0     | 82       | 82             |
| 0.25    | 76       | 76             |
| 0.5     | 69       | 69             |
| 1.0     | 53       | 53             |
| 2.0     | 15       | 19             |

Small difference for $\epsilon=2.0$.


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
