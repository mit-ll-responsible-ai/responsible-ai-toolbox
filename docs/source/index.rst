.. tip::

   🎓 Using the responsible AI toolbox for your research project? `Cite us <https://arxiv.org/abs/2201.05647>`_!


===========================================================
Welcome to the documentation for the responsible AI toolbox
===========================================================

`rai_toolbox` is designed to enable methods for evaluating and enhancing both the 
robustness and the explainability of artificial intelligence (AI) and machine 
learning (ML) models in a way that is scalable and that 
composes naturally with other popular ML frameworks.

A key design principle of `rai_toolbox` is that it adheres strictly to the APIs 
specified by the `PyTorch <https://pytorch.org/>`_ machine learning framework.
For example, `rai_toolbox` frames the process of solving for an adversarial
perturbation solely in terms of the `torch.nn.Optimizer` and `torch.nn.Module` APIs.
This makes it trivial to leverage other libraries and frameworks from the PyTorch
ecosystem to bolster your responsible AI R&D. For instance, one can naturally leverage
`rai_toolbox` together with `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ to
perform distributed adversarial training.

To see `rai_toolbox` in action, please refer to the `examples/ <https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/tree/main/examples>`_
and `experiments/ <https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/tree/main/experiments>`_ sections of our repository.



Learning about the responsible AI toolbox
=====================================

Our docs are divided into four sections: Tutorials, How-Tos, Explanations, and 
Reference.

If you want to get a bird's-eye view of what `rai_toolbox` is all about, or if you are 
completely new to executing adversarial or explainable AI workflows,
check out our **Tutorials**. For folks who are savvy responsible AI developers,
our **How-Tos** and **Reference** materials can help acquaint you with the 
unique capabilities that are offered by the toolbox. Finally, **Explanations** provide 
readers with taxonomies, design principles, recommendations, and other articles that 
will enrich their understanding of `rai_toolbox`.

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
