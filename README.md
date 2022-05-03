# Responsible AI Toolbox

The rAI-toolbox is designed to enable methods for evaluating and enhancing both the 
robustness and the explainability of AI models in a way that is scalable and that 
composes naturally with other popular ML frameworks.

A key design principle of the rAI-toolbox is that it adheres strictly to the APIs 
specified by the [PyTorch](https://pytorch.org/) machine learning framework.
For example, the rAI-toolbox frames the process of solving for an adversarial
perturbation solely in terms of the `torch.nn.Optimizer` and 
`torch.nn.Module` APIs. This makes it trivial to leverage other libraries and 
frameworks from the PyTorch ecosystem to bolster your responsible AI R&D. For 
instance, one can naturally leverage the rAI-toolbox together with
[PyTorch Lightning](https://www.pytorchlightning.ai/) to perform distributed 
adversarial training.

To see the rAI-toolbox in action, please refer to [`examples/`](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/tree/main/examples) and [`experiments/`](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/tree/main/experiments).


## Installation

To install the basic toolbox, run:

```console
pip install rai-toolbox
```

To include our "mushin" capabilities, which leverage [PyTorch Lightning](https://www.pytorchlightning.ai/) and [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) for enhanced boilerplate-free ML, run:

```console
pip install rai-toolbox[mushin]
```

## Citation

Using `rai_toolbox` for your research? Please cite the following publication:

```
@article{soklaski2022tools,
  title={Tools and Practices for Responsible AI Engineering},
  author={Soklaski, Ryan and Goodwin, Justin and Brown, Olivia and Yee, Michael and Matterer, Jason},
  journal={arXiv preprint arXiv:2201.05647},
  year={2022}
}
```


## Contributing

If you would like to contribute to this repo, please refer to our `CONTRIBUTING.md` document.



## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

- Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
- SPDX-License-Identifier: MIT

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

The software/firmware is provided to you on an As-Is basis.
