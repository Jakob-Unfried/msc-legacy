# msc-legacy
Code developed during the course of my masters thesis.
This is a legacy version and reflects the version that was used for the numerical data in the thesis.


## Overview

### autodiff
    
  Various custom autodiff formulae 
  
### figures
  + plots
  + raw_data
  + plotted_data
  
  Data processing and plotting
  
### grad_tn
  
  Code for tensornetwork contractions, the Ising model, ...
  
### jax_optimise

  Optimisation library supporting complex cost functions of
  various input structures.
  Currently implemented: only L-BFGS algorithm.
  Use `jax_optimise.main.maximise` and `jax_optimise.main.minimise`
  
### msc_projects

  Simulation files
  
### utils

  Utility functions
  

## Not included 
(to avoid licensing / plagiarism issues)
- utils/jax_utils/ncon_jax.py

  This is an adaption of ncon.py shipped with [TensorTrace](http://www.tensortrace.com) to use `jax.numpy` as a backend.
  Happy to provide pointers on how to do this on request.
  
- figures/linear_pred.py
