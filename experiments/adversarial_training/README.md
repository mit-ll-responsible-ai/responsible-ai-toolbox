# Evalulate Robustness of Adversarial Training

Adversarial Training is about solving for model parameters by evaluating against the worst-case loss:

$$
\hat{\theta} = \min_\theta \mathbb{E}_{(x,y) \sim D} \left[ \max\limits_{\|\delta\|_p<\epsilon} \mathcal{L}\left(f_\theta(x+\delta), y\right) \right]
$$

The code using `rAI-toolbox` boils down to simply running the following training loop:

```python
solve_perturbation = # define function to perturb the data
model = # load model
opt = Optim(model.parameters(), ...) # define optimizer
for epochs in range(num_epochs):
    for data, target in train_dataloader:
        perturbed_data = solve_perturbation(model, data, target, criterion)

        output = model(perturbed_data)
        loss = criterion(output, target)

        # Update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

```

## Implementation

The code is organized as follows:

```
adversarial_training
├── configs.py :  hydra-zen configurations
├── solver.py  :  The LightningModule
└── train.py   :  Main function to train
```

### Training

Here we train a ResNet-50 on CIFAR-10 with $p=2$ and $\epsilon=1$. This should reproduce the same model available from [MadryLab](https://github.com/MadryLab/robustness).  To train simply execute `train.py`:

```bash
>> python train.py
```


## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

© 2023 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
