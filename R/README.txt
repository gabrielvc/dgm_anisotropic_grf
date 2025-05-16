This folder contains all the R code used in the project.

./data/ : Contains the code used to prepare the data used in the sea surface anomaly numerical experiment.

./fine_tuning/ : Contains the code to generate the samples used to finetune our DGM in the two cases considered in the paper: 
    ./fine_tuning/global_anisotropy/ : Contains the code for the case where GRFs with constant anistropy parameters (across space) are considered
    ./fine_tuning/localized_prior/ : Contains the code for the case where GRFs related to a subpart of our original prior are considered

./grf_prior/ :  Contains the code to generate the samples from our GRF prior (used to train the DGM prior)

./packages/ : Contains the source code of R packages used in the project
    ./packages/NSGP/ : Contains the source code of the R package NSGP, developped by the authors, and used to sample GRFs (using the finite element approach) and to run Random Walk Metropolis Hastings MCMC in the case where constant anistropy parameters are considered
    ./packages/BayesNSGP-master/ : Contains the source code from the R package BayesNSGP (https://CRAN.R-project.org/package=BayesNSGP) which was pulled from the CRAN github repository (https://github.com/cran/BayesNSGP), and slightly modified by the authors to accomodate the prior used in the project. 
    
./vecchia/ : Contains the code used to run the VMCMC approach described in the paper.

./SSTA/ : Contains the code to compute the scores in the SSTA numerical experiments

The following R packages are necessary to run the code: 
  * From the CRAN: Matrix, fields, RcppEigen, rhdf5, RTriangle, tictoc, terra, scoringRules
  * From other sources : gstlearn (https://gstlearn.org/?page_id=48)
  * From the authors: NSGP, BayesNSGP (which source files are provided)
