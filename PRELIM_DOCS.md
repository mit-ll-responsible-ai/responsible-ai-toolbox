## rai_toolbox.datasets
- Provides torchvision-style datasets for CIFAR10-C and CIFAR100-C


### Usage

```python
>>> from rai_toolbox.datasets import CIFAR10C
>>> dataset = CIFAR10C("~/.torch/data/cifar/", corruption="brightness", severity=5, download=True)
>>> img, target = dataset[100]
>>> dataset.classes[target]
'deer'
>>> img
```
![image](https://user-images.githubusercontent.com/29104956/156213744-662b1029-a885-4f4f-97e8-618d576baa80.png)



## rai_toolbox.augmix


AugMix proposes a method for augmenting data using a mixture of "chains" of composed augmentations. This method appears to yield surprising improvements in robustness (i.e. accuracy) and uncertainty calibration to out-of-distribution data.

An implementation of the AugMix augmentation method specified in:

    Hendrycks, Dan, et al. "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty." International Conference on Learning Representations. 2019.

with reference implementation from https://github.com/google-research/augmix

The AugMix process is depicted in the [Figure 3 of the AugMix paper](https://arxiv.org/pdf/1912.02781.pdf), included below, the turtle picture is processed by three separate augmentation chains, which are then combined with convex coefficients. This augmented image is then mixed with the original image to produce the augmixed turtle. 


![image](https://user-images.githubusercontent.com/29104956/156213744-662b1029-a885-4f4f-97e8-618d576baa80.png)


### Usage 

`augment_and_mix` is application/framework agnostic.
E.g. it need not be used specifically with PyTorch for image processing.
In the following code snippet, we simply take a NumPy array and apply augmentations that, respectively, either add one to each number or multiply each number by 10.

```python
from rai_toolbox.augmentations.augmix import augment_and_mix
import numpy as np
```

```python
>>> data = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

>>> augment_and_mix(
...     data,
...     process_fn=lambda x: x,
...     augmentations=[lambda x: x + 1, lambda x: x * 10],
...     aug_chain_depth=4,
...     num_aug_chains=1,
...     beta_params=(1.0, 0.0001), # original data makes no contribution
...     dirichlet_params=1.0,
... )
array([ 21.,  31.,  41.,  51.,  61.,  71.,  81.,  91., 101., 111.])
```

see that, in this particular instance, the augmentation chain that was formed was `add_one ∘ mul_ten ∘ add_one ∘ add_one`.

### Replicating the results of Hendrycks et al.

There are also torchvision-style transforms for using AugMix and also for doing forked-processing,
e.g. for a consistency loss

The following code creates the train-time transform (appropriate for CIFAR10 images) that was used by Hendrycks et al. to train their model against image triplets (clean, augmix, augmix) using the Jensen-Shannon divergence consistency loss.

```python
from functools import partial

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from rai_toolbox.augmentations.augmix.augmentations import (
    CORRUPTION_OVERLAP,
    all_augmentations,
)
from rai_toolbox.augmentations import AugMix, Fork

AUG_LEVEL = 3
primitives = [
    partial(aug, level=AUG_LEVEL)
    for aug in all_augmentations
    if aug not in CORRUPTION_OVERLAP
]

pre_process = Compose(
    transforms=[
        ToTensor(),
        Normalize(
            mean=[0.5] * 3,
            std=[0.5] * 3,
        ),
    ]
)


augmix = AugMix(process_fn=pre_process, augmentations=primitives)

train_transform = Compose(
    transforms=[
        RandomHorizontalFlip(),
        RandomCrop(32, padding=4),
        Fork(pre_process, augmix, augmix),
    ]
)
```

And thus loading CIFAR10 images via the dataset:

```python
from torchvision.datasets import CIFAR10

data = CIFAR10(
    "/home/username/.torch/data/cifar/",
    transform=train_transform,
    train=True,
)
```

will yield image-triplets like:

![image](https://user-images.githubusercontent.com/29104956/156213806-f1cbdb9d-f784-4339-861b-879a1d1c6d69.png)


This triplet can then be processed using the consistency loss

```python
from rai_toolbox.losses import jensen_shannon_divergence
```


## rai_toolbox.fourier

Based off of [A Fourier Perspective on Model Robustness in Computer Vision](https://arxiv.org/abs/1906.08988)

2D Fourier-basis perturbations can be used to probe a model's response to periodic noise of various wavelengths and orientations.

Figure 3 from the [aforementioned paper](https://arxiv.org/abs/1906.08988) displays Fourier-susceptibility heat maps for three computer vision models:

![image](https://user-images.githubusercontent.com/29104956/156213847-ca3eb05d-dc5d-4852-bd81-c1023ed3b586.png)


### Usage
#### Training with Fourier Perturbations
Fourier perturbations are included as a standard data transform.
E.g. the following dataset

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor

from rai_toolbox.augmentations import FourierPerturbation

# Fourier perturbations expect to operate on the [0, 1] domain.
# Subsequent data normalization should be included after the perturbation.
data = CIFAR10(
    "/home/username/.torch/data/cifar/",
    transform=Compose([ToTensor(), FourierPerturbation((32, 32), norm_scale=4.0)]),
    train=False,
)
```

Will produce images like:

![image](https://user-images.githubusercontent.com/29104956/156213873-c4bb9432-e680-4d35-bf29-62bdaa63ae2a.png)

#### Generating a Performance Heatmap Over Fourier Perturbations

Let's produce a heatmap of classification errors for a model operating on (a random subset) CIFAR10, perturbed by the various Fourier bases.

```python
import numpy as np
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchmetrics import Accuracy

from rai_toolbox.augmentations.fourier import create_heatmaps

model = # load your model

data = CIFAR10("/home/username/.torch/data/cifar/", transform=ToTensor(), train=False)

# random subset of 100 test images
sampled_indices = np.random.choice(
    np.arange(len(data)),
    replace=False,
    size=100,
)
data = Subset(data, sampled_indices)
# The inner-loop will be very fast; it is not advised to use workers
# here. They will slow things down.
loader = DataLoader(data, shuffle=False, batch_size=100, num_workers=0, pin_memory=True)

results = create_heatmaps(
    loader,
    (32, 32),
    model=model,
    basis_norm=4.0,
    rand_flip_per_channel=True,
    # Basis-perturbation is expected to occur on [0, 1] domain.
    # Data normalization occurs after, and is applied to the entire
    # perturbed batch.
    post_pert_batch_transform=Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    device="cuda:0",
    # Metrics are any callable that takes in (pred-logits, targets)
    metrics=dict(accuracy=Accuracy()),
)
```

Visualizing the results

```python
# Populating classification-error Fourier 
# susceptibility heatmap
import matplotlib.pyplot as plt

accuracy_heatmap = np.zeros((32, 32))

for p, ps, datum in results["accuracy"]:
    accuracy_heatmap[p] = datum.item()
    accuracy_heatmap[ps] = datum.item()

fig, ax = plt.subplots()

# plotting error instead of accuracy
ax.imshow(1 - accuracy_heatmap, vmin=0, vmax=1)
```
![image](https://user-images.githubusercontent.com/29104956/156213886-5623802c-2c55-4621-812d-b50963942921.png)


## rai_toolbox.perturbation.projected_gradient

Based on [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
for calculating adversarial perturbation via Projected Gradient Descent/Ascent (PGD).

### Usage

```python
import torch
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from rai_toolbox.perturbation import projected_gradient, random_restart, L2Step

model = # load your model

dataset = CIFAR10("/path/to/data", transform=ToTensor(), train=False)

dl = DataLoader(dataset, batch_size=10, num_workers=4, pin_memory=True)
img, target = next(iter(dl))

with torch.no_grad():
    acc_before = (model(img).argmax(-1) == target).float()

# define the norm for the projection step
l2_norm_step = L2Step()
img_pg, loss_pg = projected_gradient(model, img, target, epsilon=1.0, pg_step=l2_norm_step)

with torch.no_grad():
    acc_after = (model(img_pg).argmax(-1) != target).float()
success = (acc_after == acc_before)
print(f"Successfull {sum(success)} out of 10 images")

# Or if you want to do random restarts
random_restart_pg = random_restart(pg, repeats=3)
img_pg_restarts, loss_pg_restarts = random_restart_pg(model, img, target, epsilon=1.0, pg_step=l2_norm_step)

with torch.no_grad():
    acc_after = (model(img_pg_restarts).argmax(-1) != target).float()
success = (acc_after == acc_before)
print(f"Successful with random restarts {sum(success)} out of 10 images.")
```

### Replicating MadryLab/robustness results

See [experiment/pgd](.experiment/pgd).
