.. tip::

   🎓 Using the responsible AI toolbox for your research project? `Cite us <https://arxiv.org/abs/2201.05647>`_!


===========================================================
Welcome to the documentation for the responsible AI toolbox
===========================================================

`rai_toolbox` is designed to enable methods for evaluating and enhancing both the 
robustness and the explainability of AI models in a way that is scalable and that 
composes naturally with other popular ML frameworks.

A key design principle of the rAI-toolbox is that it adheres strictly to the APIs 
specified by the PyTorch machine learning framework. For example, the rAI-toolbox frames 
adversarial training workflows solely in terms of the `torch.nn.Optimizer` and 
`torch.nn.Module` APIs. This makes it trivial to leverage other libraries and 
frameworks from the PyTorch ecosystem to bolster your responsible AI R&D. For 
instance, one can naturally leverage the rAI-toolbox together with
[PyTorch Lightning](https://www.pytorchlightning.ai/) to perform distributed 
adversarial training.

To see `rai_toolbox` in action, please refer to [`examples/`](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/tree/main/examples) and [`experiments/`](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/tree/main/experiments).



Learning About responsible AI toolbox
=====================================

Our docs are divided into four sections: Tutorials, How-Tos, Explanations, and 
Reference.

If you want to get a bird's-eye view of what hydra-zen is all about, or if you are 
completely new to Hydra, check out our **Tutorials**. For folks who are savvy Hydra 
users, our **How-Tos** and **Reference** materials can help acquaint you with the 
unique capabilities that are offered by hydra-zen. Finally, **Explanations** provide 
readers with taxonomies, design principles, recommendations, and other articles that 
will enrich their understanding of hydra-zen and Hydra.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   how_tos
   explanation
   api_reference
   changes
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
