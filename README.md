## Bayesian Optimization with High Dimensional Outputs

This is the experimental code repository for the paper [Bayesian Optimization with High Dimensional Outputs](https://arxiv.org/abs/2106.12997) (NeurIPS 2021) by Wesley Maddox, Max Balandat, Andrew Gordon Wilson, Eytan Bakshy. 

## NOTE

This repository contains experimental code for reproducibility, but we would strongly suggest that you use the `botorch.models.KroneckerMultiTaskGP` and `botorch.models.HigherOrderGP` model classes in [BoTorch](https://botorch.org).
The Kronecker linear algebra has itself been built into GPyTorch as well.

You can see these tutorials with the [HOGP](https://botorch.org/tutorials/composite_bo_with_hogp) and the [MTGP](https://botorch.org/tutorials/composite_mtbo) respectively.

Note that we cannot at this point release the SCBO codebase as well.

## Citation

```
@inproceedings{maddox2021bayesian,
  title={Bayesian Optimization with High-Dimensional Outputs},
  author={Maddox, Wesley and Balandat, Maximilian and Wilson, Andrew Gordon and Bakshy, Eytan},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

## Experimental comments

To use, please install botorch master, gpytorch master.

For the CBO experiments on Hartmann-5DEmbedding, you need to install:
https://github.com/facebookresearch/ContextualBO

For the optics + HOGP experiments, you need to install: 
https://github.com/dmitrySorokin/interferobotProject

### Main Text Figures

Figure 2: contextualbo_experiments/post_timing.py

Figure 3: contextualbo_experiments/contextual_full.py

Figure 4: mtgp_experiments/constrained_mobo.py (use --problem={c2dtlz2,osy}

Figure 5,6: not provided

Figure 7 a-c: hogp_experiments/hogp_composite_function.py (use --problem={environmental, pde, maveric1})

Figure 7d: hogp_experiments/hogp_optics.py (use --problem={optics})

### Appendix Figures

A.1: notebooks/hogp_example_workbook.ipynb

A.2: notebooks/matherons_rule_example_plot.ipynb

A.3: contextualbo_experiments/post_timing.py

A.4: contextualbo_experiments/contextual_full.py

A.5: contextualbo_experiments/cbo_experiment.py

A.6: not provided, collected from timing logs of SCBO runs

A.7-9: notebooks/hogp_output_plotting.ipynb


