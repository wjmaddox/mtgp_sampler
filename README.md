## Bayesian Optimization with High Dimensional Outputs

To use, please install botorch master, gpytorch master.

For the CBO experiments on Hartmann-5DEmbedding, you need to install:
https://github.com/facebookresearch/ContextualBO

For the optics + HOGP experiments, you need to install: 
https://github.com/dmitrySorokin/interferobotProject

We are unable to provide the code for the SCBO experiments at this point in time due to licensing issues on
the side of the authors of that code.

We hope to make these available for the camera ready.

### Main Text

Figure 2: contextualbo_experiments/post_timing.py

Figure 3: contextualbo_experiments/contextual_full.py

Figure 4: mtgp_experiments/constrained_mobo.py (use --problem={c2dtlz2,osy}

Figure 5,6: not provided

Figure 7 a-c: hogp_experiments/hogp_composite_function.py (use --problem={environmental, pde, maveric1})

Figure 7d: hogp_experiments/hogp_optics.py (use --problem={optics})

### Appendix

A.1: notebooks/hogp_example_workbook.ipynb

A.2: notebooks/matherons_rule_example_plot.ipynb

A.3: contextualbo_experiments/post_timing.py

A.4: contextualbo_experiments/contextual_full.py

A.5: contextualbo_experiments/cbo_experiment.py

A.6: not provided, collected from timing logs of SCBO runs

A.7-9: notebooks/hogp_output_plotting.ipynb


