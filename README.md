# paper-cme-tt
This is the repository for the code of the paper "Tensor-train approximation of the chemical master equation and its application for parameter inference"


Requirements:
- numpy (>=1.18.1)
- ttpy (Intel MKL is recommended)
- cython
- scipy
- matplotlib
- tensorflow (>=2.0)
- pickle 

The scripts that produce the results from the paper are:
- simplegene_tt_convergence.py for the time convergence of the seir model.
- model_seir_tt.py for the solution of the SEIR model.
- seir_filtering.py for the filtering and smoothing of the SEIR model.
- simplegene_tt_convergence.py for the simple gene model time stepping convergence.
- simple_gene_tt_4_param_projection.py for the Bayesian inference of the simple gene model.
- simple_gene_tt_4_param_sweep.py for inference using the simple gene for different hyperparameters.
- 3stage_param_projection.py for the 3 stage gene expression experiment.
- model_SEIQR_param.py SEIQR experiment.



