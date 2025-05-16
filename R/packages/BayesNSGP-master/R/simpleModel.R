
inverseEigen_custom <- nimble::nimbleFunction(
  run = function( eigen_comp1 = double(1), eigen_comp2 = double(1),
                  eigen_comp3 = double(1), which_Sigma = double(0) ) {
    returnType(double(1))
    
    ppi=3.1415926535897932384626433
    rotAngle <- ppi * exp(eigen_comp3)/(1 + exp(eigen_comp3))
    
    Gam11 <- cos(rotAngle)
    Gam22 <- cos(rotAngle)
    Gam12 <- -sin(rotAngle)
    Gam21 <- sin(rotAngle)
    
    Lam1 <- (eigen_comp1)
    Lam2 <- (eigen_comp2)
    
    if( which_Sigma == 1 ){ # Return Sigma11
      return( Gam11^2*Lam1 + Gam12^2*Lam2 )
    }
    if( which_Sigma == 2 ){ # Return Sigma22
      return( Gam21^2*Lam1 + Gam22^2*Lam2 )
    }
    if( which_Sigma == 3 ){ # Return Sigma12
      return( Gam11*Gam21*Lam1 + Gam12*Gam22*Lam2 )
    }
    
    stop('Error in inverseEigen function')  ## prevent compiler warning
    return(numeric(10))                     ## prevent compiler warning
    
  }
)



inverseEigen_ganiso <- nimble::nimbleFunction(
  run = function( eigen_comp1 = double(1), eigen_comp2 = double(1),
                  eigen_comp3 = double(1), which_Sigma = double(0) ) {
    returnType(double(1))
    
    ppi=3.1415926535897932384626433
    rotAngle <- eigen_comp3
    
    Gam11 <- cos(rotAngle)
    Gam22 <- cos(rotAngle)
    Gam12 <- -sin(rotAngle)
    Gam21 <- sin(rotAngle)
    
    Lam1 <- (eigen_comp1)
    Lam2 <- (eigen_comp2)
    
    if( which_Sigma == 1 ){ # Return Sigma11
      return( Gam11^2*Lam1 + Gam12^2*Lam2 )
    }
    if( which_Sigma == 2 ){ # Return Sigma22
      return( Gam21^2*Lam1 + Gam22^2*Lam2 )
    }
    if( which_Sigma == 3 ){ # Return Sigma12
      return( Gam11*Gam21*Lam1 + Gam12*Gam22*Lam2 )
    }
    
    stop('Error in inverseEigen function')  ## prevent compiler warning
    return(numeric(10))                     ## prevent compiler warning
    
  }
)


#' 
#' Simpler Model
#' 
#' @export
#' 
nsgpModelSimple <- function( tau_model   = "constant",
                       sigma_model = "constant",
                       Sigma_model = "constant",
                       mu_model    = "constant",
                       likelihood  = "fullGP",
                       coords,
                       data,
                       constants = list(),
                       const_params=list(),
                       monitorAllSampledNodes = TRUE,
                       constInit=FALSE,
                       sgvOrdering=NULL,
                       ... ) {
  
  
  
  
  ##============================================
  ## Models for tau
  ##============================================
  if(!is.null(const_params$tau)){
    tau_model_list <- list(
      constant = list(
        ## 1. tau_HP1          Standard deviation for the log-linear standard deviation
        ## 2. delta            Scalar; represents the standard deviation (constant over the domain)
        ## 3. ones             N-vector of 1's
        code = quote({
          log_tau_vec[1:N] <- log(sqrt(delta))*ones[1:N]
        }),
        constants_needed = c("ones"),
        inits = list(delta = const_params$tau**2)
      ))
  }else{
  tau_model_list <- list(
    constant = list(
      ## 1. tau_HP1          Standard deviation for the log-linear standard deviation
      ## 2. delta            Scalar; represents the standard deviation (constant over the domain)
      ## 3. ones             N-vector of 1's
      code = quote({
        log_tau_vec[1:N] <- log(sqrt(delta))*ones[1:N]
        delta ~ dunif(0, tau_HP1)
      }),
      constants_needed = c("ones", "tau_HP1"),
      inits = list(delta = quote(tau_HP1/10))
    ),
    fixed = list(
      ## 1. tau_HP1          Standard deviation for the log-linear standard deviation
      ## 2. delta            Scalar; represents the standard deviation (constant over the domain)
      ## 3. ones             N-vector of 1's
      code = quote({
        log_tau_vec[1:N] <- log(tau_HP1)*ones[1:N]
      }),
      constants_needed = c("ones", "tau_HP1")
    ),
    logLinReg = list(
      ## 1. X_tau            N x p_tau design matrix; leading column of 1's with (p_tau - 1) other covariates
      ## 2. tau_HP1          Standard deviation for the log-linear regression coefficients
      ## 3. p_tau            Number of design columns
      ## 4. delta            Vector of length p_tau; represents log-linear regression coefficients
      code = quote({
        log_tau_vec[1:N] <- X_tau[1:N,1:p_tau] %*% delta[1:p_tau]
        for(l in 1:p_tau){
          delta[l] ~ dnorm(0, sd = tau_HP1)
        }
        tau_constraint1 ~ dconstraint( max(abs(log_tau_vec[1:N])) < maxAbsLogSD )
      }),
      constants_needed = c("X_tau", "p_tau", "tau_HP1", "maxAbsLogSD"),
      inits = list(delta = quote(rep(0, p_tau))),
      constraints_needed = c('tau_constraint1')
    ),
    
    approxGP = list(
      ## 1. tau_HP1          Gaussian process standard deviation
      ## 2. tau_HP2          Gaussian process mean
      ## 3. tau_HP3          Gaussian process range
      ## 4. tau_HP4          Gaussian process smoothness
      ## 5. ones             N-vector of 1's
      ## 6. tau_cross_dist   N x p_tau matrix of inter-point Euclidean distances, obs. coords vs. knot locations
      ## 7. tau_knot_dist    p_tau x p_tau matrix of inter-point Euclidean distances, knot locations
      ## 8. p_tau            Number of knot locations
      code = quote({
        log_tau_vec[1:N] <- tauGP_mu*ones[1:N] + tauGP_sigma*Pmat_tau[1:N,1:p_tau] %*% w_tau[1:p_tau]
        Pmat_tau[1:N,1:p_tau] <- matern_corr(tau_cross_dist[1:N,1:p_tau], tauGP_phi, tau_HP2)
        Vmat_tau[1:p_tau,1:p_tau] <- matern_corr(tau_knot_dist[1:p_tau,1:p_tau], tauGP_phi, tau_HP2)
        w_tau_mean[1:p_tau] <- 0*ones[1:p_tau]
        w_tau[1:p_tau] ~ dmnorm( mean = w_tau_mean[1:p_tau], prec = Vmat_tau[1:p_tau,1:p_tau] )
        # Hyperparameters
        tauGP_mu ~ dnorm(0, sd = tau_HP1)
        tauGP_phi ~ dunif(0, tau_HP3) # Range parameter, GP
        tauGP_sigma ~ dunif(0, tau_HP4) # SD parameter, GP
        # Constraint
        tau_constraint1 ~ dconstraint( max(abs(log_tau_vec[1:N])) < maxAbsLogSD )
      }),
      constants_needed = c("ones", "tau_knot_coords", "tau_cross_dist", "tau_knot_dist", 
                           "p_tau", "tau_HP1", "tau_HP2", "tau_HP3", "tau_HP4", "maxAbsLogSD"),
      inits = list(
        w_tau = quote(rep(0, p_tau)),
        tauGP_mu = quote(0),
        tauGP_phi = quote(tau_HP3/100),
        tauGP_sigma = quote(tau_HP4/100)
      ),
      constraints_needed = c('tau_constraint1')
      
    )
  )
  }
  
  ##============================================
  ## Models for sigma
  ##============================================
  if(!is.null(const_params$sigma)){
    sigma_model_list <- list(
      constant = list(
        ## 3. ones             N-vector of 1's
        code = quote({
          log_sigma_vec[1:N] <- log(sqrt(alpha))*ones[1:N]
        }),
        constants_needed = c("ones"),
        inits = list(alpha = const_params$sigma**2)
      )
    )
  }else{
  sigma_model_list <- list(
    constant = list(
      ## 1. sigma_HP1        Standard deviation for the log-linear standard deviation
      ## 2. alpha            Scalar; represents the standard deviation (constant over the domain)
      ## 3. ones             N-vector of 1's
      code = quote({
        log_sigma_vec[1:N] <- log(sqrt(alpha))*ones[1:N]
        alpha ~ dunif(0, sigma_HP1)
      }),
      constants_needed = c("ones", "sigma_HP1"),
      inits = list(alpha = quote(sigma_HP1/10))
    ),
    fixed = list(
      ## 1. sigma_HP1        Standard deviation for the log-linear standard deviation
      ## 2. alpha            Scalar; represents the standard deviation (constant over the domain)
      ## 3. ones             N-vector of 1's
      code = quote({
        log_sigma_vec[1:N] <- log(sigma_HP1)*ones[1:N]
      }),
      constants_needed = c("ones", "sigma_HP1")
    ),
    logLinReg = list(
      ## 1. X_sigma          N x p_sigma design matrix; leading column of 1's with (p_sigma - 1) other covariates
      ## 2. sigma_HP1        Standard deviation for the log-linear regression coefficients
      ## 3. p_sigma          Number of design columns
      ## 4. alpha            Vector of length p_sigma; represents log-linear regression coefficients
      code = quote({
        log_sigma_vec[1:N] <- X_sigma[1:N,1:p_sigma] %*% alpha[1:p_sigma]
        for(l in 1:p_sigma){
          alpha[l] ~ dnorm(0, sd = sigma_HP1)
        }
        # Constraint
        sigma_constraint1 ~ dconstraint( max(abs(log_sigma_vec[1:N])) < maxAbsLogSD )
      }),
      constants_needed = c("X_sigma", "p_sigma", "sigma_HP1", "maxAbsLogSD"),
      inits = list(alpha = quote(rep(0, p_sigma))),
      constraints_needed = c('sigma_constraint1')
    ),
    
    approxGP = list(
      ## 1. sigma_HP1        Gaussian process standard deviation
      ## 2. sigma_HP2        Gaussian process mean
      ## 3. sigma_HP3        Gaussian process range
      ## 4. sigma_HP4        Gaussian process smoothness
      ## 5. ones             N-vector of 1's
      ## 6. sigma_cross_dist N x p_sigma matrix of inter-point Euclidean distances, obs. coords vs. knot locations
      ## 7. sigma_knot_dist  p_sigma x p_sigma matrix of inter-point Euclidean distances, knot locations
      ## 8. p_sigma          Number of knot locations
      code = quote({
        log_sigma_vec[1:N] <- sigmaGP_mu*ones[1:N] + sigmaGP_sigma*Pmat_sigma[1:N,1:p_sigma] %*% w_sigma[1:p_sigma]
        Pmat_sigma[1:N,1:p_sigma] <- matern_corr(sigma_cross_dist[1:N,1:p_sigma], sigmaGP_phi, sigma_HP2)
        Vmat_sigma[1:p_sigma,1:p_sigma] <- matern_corr(sigma_knot_dist[1:p_sigma,1:p_sigma], sigmaGP_phi, sigma_HP2)
        w_sigma_mean[1:p_sigma] <- 0*ones[1:p_sigma]
        w_sigma[1:p_sigma] ~ dmnorm( mean = w_sigma_mean[1:p_sigma], prec = Vmat_sigma[1:p_sigma,1:p_sigma] )
        # Hyperparameters
        sigmaGP_mu ~ dnorm(0, sd = sigma_HP1)
        sigmaGP_phi ~ dunif(0, sigma_HP3) # Range parameter, GP
        sigmaGP_sigma ~ dunif(0, sigma_HP4) # SD parameter, GP
        # Constraint
        sigma_constraint1 ~ dconstraint( max(abs(log_sigma_vec[1:N])) < maxAbsLogSD )
      }),
      constants_needed = c("ones", "sigma_knot_coords", "sigma_cross_dist", "sigma_knot_dist", 
                           "p_sigma", "sigma_HP1", "sigma_HP2", "sigma_HP3", "sigma_HP4", "maxAbsLogSD"),
      inits = list(
        w_sigma = quote(rep(0, p_sigma)),
        sigmaGP_mu = quote(0),
        sigmaGP_phi = quote(sigma_HP3/100),
        sigmaGP_sigma = quote(sigma_HP4/100)),
      constraints_needed = c('sigma_constraint1')
    )
  )
  }
  
  ##============================================
  ## Models for Sigma
  ##============================================
  Sigma_model_list <- list(
    
    constant = list(
      ## 1. ones                N-vector of 1's
      ## 2. Sigma_HP1           Upper bound for the eigenvalues
      ## 3. Sigma_coef{1,2,3}   Vectors of length p_Sigma; represents the anisotropy components
      code = quote({
        
        Sigma11[1:N] <- ones[1:N]*(Sigma_coef1*cos(Sigma_coef3)*cos(Sigma_coef3) + Sigma_coef2*sin(Sigma_coef3)*sin(Sigma_coef3))
        Sigma22[1:N] <- ones[1:N]*(Sigma_coef2*cos(Sigma_coef3)*cos(Sigma_coef3) + Sigma_coef1*sin(Sigma_coef3)*sin(Sigma_coef3))
        Sigma12[1:N] <- ones[1:N]*(Sigma_coef1*cos(Sigma_coef3)*sin(Sigma_coef3) - Sigma_coef2*cos(Sigma_coef3)*sin(Sigma_coef3))
        
        Sigma_coef1 ~ dunif(0, Sigma_HP1[1]) # phi1
        Sigma_coef2 ~ dunif(0, Sigma_HP1[1]) # phi2
        Sigma_coef3 ~ dunif(0, 1.570796)  # eta --> 1.570796 = pi/2
        
      }),
      constants_needed = c("ones", "Sigma_HP1"),
      inits = list(
        Sigma_coef1 = quote(Sigma_HP1[1]/4),
        Sigma_coef2 = quote(Sigma_HP1[1]/4),
        Sigma_coef3 = 0.7853982 # pi/4
      )
    ),
    constantIso = list( # Isotropic version of 
      ## 1. ones                 N-vector of 1's
      ## 2. Sigma_HP1            Standard deviation for the anisotropy components
      ## 3. Sigma_coef{1,2,3}    Vectors of length p_Sigma; represents the anisotropy components
      code = quote({
        Sigma11[1:N] <- ones[1:N]*Sigma_coef1
        Sigma22[1:N] <- ones[1:N]*Sigma_coef1
        Sigma12[1:N] <- ones[1:N]*0
        
        Sigma_coef1 ~ dunif(0, Sigma_HP1[1]) # phi1
      }),
      constants_needed = c("ones", "Sigma_HP1"),
      inits = list( Sigma_coef1 = quote(Sigma_HP1[1]/4) )
    ),
    
    covReg = list(
      code = quote({
        ## 1. X_Sigma                N x p_Sigma design matrix; leading column of 1's with (p_Sigma - 1) other covariates
        ## 2. Sigma_HP1              Standard deviation for the covariance regression coefficients
        ## 3. p_Sigma                Number of design columns
        ## 4. gamma1, gamma2         Vectors of length p_Sigma; represents covariance regression coefficients
        ## 5. psi11, psi22, rho      Baseline covariance regression parameters
        ## 6. Sigma_HP2              Upper bound for the baseline covariance regression variances
        Sigma11[1:N] <- psi11*ones[1:N] + (X_Sigma[1:N,1:p_Sigma] %*% gamma1[1:p_Sigma])^2
        Sigma12[1:N] <- rho*sqrt(psi11*psi22)*ones[1:N] + (X_Sigma[1:N,1:p_Sigma]%*%gamma1[1:p_Sigma])*(X_Sigma[1:N,1:p_Sigma]%*%gamma2[1:p_Sigma])
        Sigma22[1:N] <- psi22*ones[1:N] + (X_Sigma[1:N,1:p_Sigma] %*% gamma2[1:p_Sigma])^2
        psi11 ~ dunif(0, Sigma_HP2[1])
        psi22 ~ dunif(0, Sigma_HP2[2])
        rho ~ dunif(-1, 1)
        for(j in 1:p_Sigma){
          gamma1[j] ~ dnorm(0, sd = Sigma_HP1[1])
          gamma2[j] ~ dnorm(0, sd = Sigma_HP1[2])
        }
        # Constraints: upper limits on eigen_comp1 and eigen_comp2
        Sigma_constraint1 ~ dconstraint( max(Sigma11[1:N]) < maxAnisoRange )
        Sigma_constraint2 ~ dconstraint( max(Sigma22[1:N]) < maxAnisoRange )
        Sigma_constraint3 ~ dconstraint( min(Sigma11[1:N]*Sigma22[1:N] - Sigma12[1:N]*Sigma12[1:N]) > minAnisoDet )
      }),
      constants_needed = c("ones", "X_Sigma", "p_Sigma", "Sigma_HP1", "Sigma_HP2", "maxAnisoRange", "minAnisoDet"),
      inits = list(
        psi11 = quote(Sigma_HP2[1]/4),
        psi22 = quote(Sigma_HP2[2]/4),
        rho = 0,
        gamma1 = quote(rep(0, p_Sigma)),
        gamma2 = quote(rep(0, p_Sigma))
      ),
      constraints_needed = c('Sigma_constraint1', 'Sigma_constraint2', 'Sigma_constraint3')
    ), 
    compReg = list(
      code = quote({
        ## 1. X_Sigma                N x p_Sigma design matrix; leading column of 1's with (p_Sigma - 1) other covariates
        ## 2. Sigma_HP1              Standard deviation for the component regression coefficients
        ## 3. p_Sigma                Number of design columns
        ## 4. Sigma_coef{1,2,3}      Vectors of length p_Sigma; represents component regression coefficients
        eigen_comp1[1:N] <- X_Sigma[1:N,1:p_Sigma] %*% Sigma_coef1[1:p_Sigma]
        eigen_comp2[1:N] <- X_Sigma[1:N,1:p_Sigma] %*% Sigma_coef2[1:p_Sigma]
        eigen_comp3[1:N] <- X_Sigma[1:N,1:p_Sigma] %*% Sigma_coef3[1:p_Sigma]
        Sigma11[1:N] <- inverseEigen(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 1)
        Sigma12[1:N] <- inverseEigen(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 3) 
        Sigma22[1:N] <- inverseEigen(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 2)
        for(j in 1:p_Sigma){
          Sigma_coef1[j] ~ dnorm(0, sd = Sigma_HP1[1])
          Sigma_coef2[j] ~ dnorm(0, sd = Sigma_HP1[1])
          Sigma_coef3[j] ~ dnorm(0, sd = Sigma_HP1[1])
        }
        # Constraints: upper limits on eigen_comp1 and eigen_comp2
        Sigma_constraint1 ~ dconstraint( max(Sigma11[1:N]) < maxAnisoRange )
        Sigma_constraint2 ~ dconstraint( max(Sigma22[1:N]) < maxAnisoRange )
        Sigma_constraint3 ~ dconstraint( min(Sigma11[1:N]*Sigma22[1:N] - Sigma12[1:N]*Sigma12[1:N]) > minAnisoDet )
      }),
      constants_needed = c("X_Sigma", "p_Sigma", "Sigma_HP1", "maxAnisoRange", "minAnisoDet"),
      inits = list(
        Sigma_coef1 = quote(c(log(maxAnisoRange/100), rep(0, p_Sigma-1))),
        Sigma_coef2 = quote(c(log(maxAnisoRange/100), rep(0, p_Sigma-1))),
        Sigma_coef3 = quote(rep(0, p_Sigma))
      ),
      constraints_needed = c('Sigma_constraint1', 'Sigma_constraint2', 'Sigma_constraint3')
    ),
    custom = list(
      code = quote({
        ## 1. X_Sigma                N x p_Sigma design matrix; leading column of 1's with (p_Sigma - 1) other covariates
        ## 2. Sigma_HP1              Standard deviation for the component regression coefficients
        ## 3. p_Sigma                Number of design columns
        ## 4. Sigma_coef{1,2,3}      Vectors of length p_Sigma; represents component regression coefficients
        eigen_comp1[1:N] <- ((Sigma_coef1/4)**2)*ones[1:N]
        eigen_comp2[1:N] <- ((Sigma_coef1*Sigma_coef2/4)**2)*ones[1:N]
        # eigen_comp3[1:N] <- inprod(X_Sigma[1:N,1:p_Sigma],Sigma_coef3[1:p_Sigma])
        eigen_comp3[1:N] <- X_Sigma[1:N,1:p_Sigma]%*%Sigma_coef3[1:p_Sigma]
        
        
        Sigma11[1:N] <- inverseEigen_custom(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 1)
        Sigma12[1:N] <- inverseEigen_custom(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 3) 
        Sigma22[1:N] <- inverseEigen_custom(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 2)
        
        Sigma_coef1 ~ dunif(0.05,0.3)
        Sigma_coef2 ~ dunif(0.1,1)
        for(j in 1:p_Sigma){
          Sigma_coef3[j] ~ dnorm(0, sd = Sigma_HP1[1])
        }
      }),
      constants_needed = c("X_Sigma", "p_Sigma", "Sigma_HP1"),
      inits = list(
        Sigma_coef1 = if(constInit) 0.1 else quote(runif(1,0.05,0.3)),
        Sigma_coef2 = if(constInit) 0.99 else quote(runif(1,0.1,1)),
        Sigma_coef3 = if(constInit) quote(rep(0,p_Sigma)) else quote(rnorm(p_Sigma,0,Sigma_HP1[1]))
      ) 
    ),
    ganiso = list(
      code = quote({
        ## 1. X_Sigma                N x p_Sigma design matrix; leading column of 1's with (p_Sigma - 1) other covariates
        ## 2. Sigma_HP1              Standard deviation for the component regression coefficients
        ## 3. p_Sigma                Number of design columns
        ## 4. Sigma_coef{1,2,3}      Vectors of length p_Sigma; represents component regression coefficients
        eigen_comp1[1:N] <- ((Sigma_coef1/4)**2)*ones[1:N]
        eigen_comp2[1:N] <- ((Sigma_coef1*Sigma_coef2/4)**2)*ones[1:N]
        eigen_comp3[1:N] <- (Sigma_coef3)*ones[1:N]
        
        Sigma11[1:N] <- inverseEigen_ganiso(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 1)
        Sigma12[1:N] <- inverseEigen_ganiso(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 3) 
        Sigma22[1:N] <- inverseEigen_ganiso(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 2)
        
        Sigma_coef1 ~ dunif(0.05,0.3)
        Sigma_coef2 ~ dunif(0.1,1)
        Sigma_coef3 ~ dunif(0,3.1415926535897932384626433)

      }),
      constants_needed = c("ones"),
      inits = list(
        Sigma_coef1 = quote(runif(1,0.05,0.3)),
        Sigma_coef2 = quote(runif(1,0.1,1)),
        Sigma_coef3 = quote(runif(1,0,3.1415926535897932384626433))
      ) 
    ),
    compRegIso = list( # Isotropic version of compReg
      code = quote({
        ## 1. X_Sigma                N x p_Sigma design matrix; leading column of 1's with (p_Sigma - 1) other covariates
        ## 2. Sigma_HP1              Standard deviation for the component regression coefficients
        ## 3. p_Sigma                Number of design columns
        ## 4. Sigma_coef{1,2,3}      Vectors of length p_Sigma; represents component regression coefficients
        eigen_comp1[1:N] <- X_Sigma[1:N,1:p_Sigma] %*% Sigma_coef1[1:p_Sigma]
        Sigma11[1:N] <- exp(eigen_comp1[1:N])
        Sigma22[1:N] <- exp(eigen_comp1[1:N])
        Sigma12[1:N] <- ones[1:N]*0
        for(j in 1:p_Sigma){
          Sigma_coef1[j] ~ dnorm(0, sd = Sigma_HP1[1])
        }
        # Constraints: upper limits on eigen_comp1
        Sigma_constraint1 ~ dconstraint( max(Sigma11[1:N]) < maxAnisoRange )
      }),
      constants_needed = c("ones", "X_Sigma", "p_Sigma", "Sigma_HP1", "maxAnisoRange"),
      inits = list(
        Sigma_coef1 = quote(c(log(maxAnisoRange/100), rep(0, p_Sigma-1)))
      ),
      constraints_needed = c('Sigma_constraint1')
    ),
    npApproxGP = list( 
      code = quote({
        ## 1. Sigma_HP1          3-vector; Gaussian process mean
        ## 2. Sigma_HP2          3-vector; Gaussian process smoothness
        ## 5. ones                   N-vector of 1's
        ## 6. dist                   N x N matrix of inter-point Euclidean distances
        ## 7. Sigma_cross_dist       N x p_Sigma matrix of inter-point Euclidean distances, obs. coords vs. knot locations
        ## 8. Sigma_knot_dist        p_Sigma x p_Sigma matrix of inter-point Euclidean distances, knot locations
        ## 9. p_Sigma                Number of knot locations
        
        Sigma11[1:N] <- inverseEigen(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 1)
        Sigma12[1:N] <- inverseEigen(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 3) 
        Sigma22[1:N] <- inverseEigen(eigen_comp1[1:N], eigen_comp2[1:N], eigen_comp3[1:N], 2)
        
        # approxGP1, approxGP2
        eigen_comp1[1:N] <- SigmaGP_mu[1]*ones[1:N] + SigmaGP_sigma[1] * Pmat12_Sigma[1:N,1:p_Sigma] %*% w1_Sigma[1:p_Sigma]
        eigen_comp2[1:N] <- SigmaGP_mu[1]*ones[1:N] + SigmaGP_sigma[1] * Pmat12_Sigma[1:N,1:p_Sigma] %*% w2_Sigma[1:p_Sigma]
        
        Pmat12_Sigma[1:N,1:p_Sigma] <- matern_corr(Sigma_cross_dist[1:N,1:p_Sigma], SigmaGP_phi[1], Sigma_HP2[1])
        Vmat12_Sigma[1:p_Sigma,1:p_Sigma] <- matern_corr(Sigma_knot_dist[1:p_Sigma,1:p_Sigma], SigmaGP_phi[1], Sigma_HP2[1])
        w12_Sigma_mean[1:p_Sigma] <- 0*ones[1:p_Sigma]
        
        w1_Sigma[1:p_Sigma] ~ dmnorm( mean = w12_Sigma_mean[1:p_Sigma], prec = Vmat12_Sigma[1:p_Sigma,1:p_Sigma] )
        w2_Sigma[1:p_Sigma] ~ dmnorm( mean = w12_Sigma_mean[1:p_Sigma], prec = Vmat12_Sigma[1:p_Sigma,1:p_Sigma] )
        
        # approxGP3
        eigen_comp3[1:N] <- SigmaGP_mu[2]*ones[1:N] + SigmaGP_sigma[2] * Pmat3_Sigma[1:N,1:p_Sigma] %*% w3_Sigma[1:p_Sigma]
        Pmat3_Sigma[1:N,1:p_Sigma] <- matern_corr(Sigma_cross_dist[1:N,1:p_Sigma], SigmaGP_phi[2], Sigma_HP2[2])
        Vmat3_Sigma[1:p_Sigma,1:p_Sigma] <- matern_corr(Sigma_knot_dist[1:p_Sigma,1:p_Sigma], SigmaGP_phi[2], Sigma_HP2[2])
        w3_Sigma_mean[1:p_Sigma] <- 0*ones[1:p_Sigma]
        w3_Sigma[1:p_Sigma] ~ dmnorm( mean = w3_Sigma_mean[1:p_Sigma], prec = Vmat3_Sigma[1:p_Sigma,1:p_Sigma] )
        
        # Hyperparameters
        for(w in 1:2){
          SigmaGP_mu[w] ~ dnorm(0, sd = Sigma_HP1[w])
          SigmaGP_phi[w] ~ dunif(0, Sigma_HP3[w]) # Range parameter, GP
          SigmaGP_sigma[w] ~ dunif(0, Sigma_HP4[w]) # SD parameter, GP
        }
        
        # Constraints: upper limits on eigen_comp1 and eigen_comp2
        Sigma_constraint1 ~ dconstraint( max(Sigma11[1:N]) < maxAnisoRange )
        Sigma_constraint2 ~ dconstraint( max(Sigma22[1:N]) < maxAnisoRange )
        Sigma_constraint3 ~ dconstraint( min(Sigma11[1:N]*Sigma22[1:N] - Sigma12[1:N]*Sigma12[1:N]) > minAnisoDet )
      }),
      constants_needed = c("ones", "Sigma_HP1", "Sigma_HP2", "Sigma_HP3", "Sigma_HP4", "maxAnisoRange", "minAnisoDet",
                           "Sigma_knot_coords", "Sigma_cross_dist", "Sigma_knot_dist", "p_Sigma"),    
      inits = list(
        w1_Sigma = quote(rep(0,p_Sigma)),
        w2_Sigma = quote(rep(0,p_Sigma)),
        w3_Sigma = quote(rep(0,p_Sigma)),
        SigmaGP_mu = quote(rep(log(maxAnisoRange/100),2)),
        SigmaGP_phi = quote(rep(Sigma_HP3[1]/100,2)),
        SigmaGP_sigma = quote(rep(Sigma_HP4[1]/100,2))
      ),
      constraints_needed = c('Sigma_constraint1', 'Sigma_constraint2', 'Sigma_constraint3')
    ),
    
    npApproxGPIso = list( 
      code = quote({
        ## 1. Sigma_HP1          3-vector; Gaussian process mean
        ## 2. Sigma_HP2          3-vector; Gaussian process smoothness
        ## 5. ones                   N-vector of 1's
        ## 6. dist                   N x N matrix of inter-point Euclidean distances
        ## 7. Sigma_cross_dist       N x p_Sigma matrix of inter-point Euclidean distances, obs. coords vs. knot locations
        ## 8. Sigma_knot_dist        p_Sigma x p_Sigma matrix of inter-point Euclidean distances, knot locations
        ## 9. p_Sigma                Number of knot locations
        
        Sigma11[1:N] <- exp(eigen_comp1[1:N])
        Sigma22[1:N] <- exp(eigen_comp1[1:N])
        Sigma12[1:N] <- ones[1:N]*0
        
        # approxGP1
        eigen_comp1[1:N] <- SigmaGP_mu[1]*ones[1:N] + SigmaGP_sigma[1] * Pmat12_Sigma[1:N,1:p_Sigma] %*% w1_Sigma[1:p_Sigma]
        
        Pmat12_Sigma[1:N,1:p_Sigma] <- matern_corr(Sigma_cross_dist[1:N,1:p_Sigma], SigmaGP_phi[1], Sigma_HP2[1])
        Vmat12_Sigma[1:p_Sigma,1:p_Sigma] <- matern_corr(Sigma_knot_dist[1:p_Sigma,1:p_Sigma], SigmaGP_phi[1], Sigma_HP2[1])
        w12_Sigma_mean[1:p_Sigma] <- 0*ones[1:p_Sigma]
        
        w1_Sigma[1:p_Sigma] ~ dmnorm( mean = w12_Sigma_mean[1:p_Sigma], prec = Vmat12_Sigma[1:p_Sigma,1:p_Sigma] )
        
        # Hyperparameters
        for(w in 1){
          SigmaGP_mu[w] ~ dnorm(0, sd = Sigma_HP1[w])
          SigmaGP_phi[w] ~ dunif(0, Sigma_HP3[w]) # Range parameter, GP
          SigmaGP_sigma[w] ~ dunif(0, Sigma_HP4[w]) # SD parameter, GP
        }
        
        # Constraints: upper limits on eigen_comp1 and eigen_comp2
        Sigma_constraint1 ~ dconstraint( max(Sigma11[1:N]) < maxAnisoRange )
      }),
      constants_needed = c("ones", "Sigma_HP1", "Sigma_HP2", "Sigma_HP3", "Sigma_HP4", "maxAnisoRange", 
                           "Sigma_knot_coords", "Sigma_cross_dist", "Sigma_knot_dist", "p_Sigma"),    
      inits = list(
        w1_Sigma = quote(rep(0,p_Sigma)),
        SigmaGP_mu = quote(rep(log(maxAnisoRange/100),1)),
        SigmaGP_phi = quote(rep(Sigma_HP3[1]/100,1)),
        SigmaGP_sigma = quote(rep(Sigma_HP4[1]/100,1))
      ),
      constraints_needed = c('Sigma_constraint1')
      
    )
  )
  
  ##============================================
  ## Models for mu
  ##============================================
  if(!is.null(const_params$mu)){
    mu_model_list <- list(
      constant = list(
        ## 1. sigma_HP1          Standard deviation for the log-linear standard deviation
        ## 2. alpha                  Scalar; represents log-linear standard deviation (constant over the domain)
        ## 3. ones                   N-vector of 1's
        code = quote({
          mu[1:N] <-beta*ones[1:N]
        }),
        constants_needed = c("ones"),
        inits = list(beta = const_params$mu))
    )
  }else{
  mu_model_list <- list(
    constant = list(
      ## 1. sigma_HP1          Standard deviation for the log-linear standard deviation
      ## 2. alpha                  Scalar; represents log-linear standard deviation (constant over the domain)
      ## 3. ones                   N-vector of 1's
      code = quote({
        mu[1:N] <-beta*ones[1:N]
        beta ~ dnorm(0, sd = mu_HP1)
      }),
      constants_needed = c("ones", "mu_HP1"),
      inits = list(beta = 0)),
    linReg = list(
      ## 1. X_mu                N x p_mu design matrix; leading column of 1's with (p_mu - 1) other covariates
      ## 2. p_mu                Number of design columns
      ## 3. beta                Vector of length p_mu; represents regression coefficients
      code = quote({
        mu[1:N] <- X_mu[1:N,1:p_mu] %*% beta[1:p_mu]
        for(l in 1:p_mu){
          beta[l] ~ dnorm(0, sd = mu_HP1)
        }
      }),
      constants_needed = c("X_mu", "p_mu", "mu_HP1"),
      inits = list(beta = quote(rep(0, p_mu)))),
    zero = list(
      ## 1. zeros               N-vector of 0's
      code = quote({
        mu[1:N] <- zeros[1:N]
      }),
      constants_needed = c("zeros"),
      inits = list()
    )
  )
  
  ##============================================
  ## Models for likelihood
  ##============================================
  likelihood_list <- list(
    fullGP = list(
      code = quote({
        Cor[1:N,1:N] <- nsCorr(dist1_sq[1:N,1:N], dist2_sq[1:N,1:N], dist12[1:N,1:N],
                               Sigma11[1:N], Sigma22[1:N], Sigma12[1:N], nu, d)
        sigmaMat[1:N,1:N] <- diag(exp(log_sigma_vec[1:N]))
        Cov[1:N, 1:N] <- sigmaMat[1:N,1:N] %*% Cor[1:N,1:N] %*% sigmaMat[1:N,1:N]
        C[1:N,1:N] <- Cov[1:N, 1:N] + diag(exp(log_tau_vec[1:N])^2)
        z[1:N] ~ dmnorm(mean = mu[1:N], cov = C[1:N,1:N])
      }),
      constants_needed = c("N", "coords", "d", "dist1_sq", "dist2_sq", "dist12", "nu"),                ## keep N, coords, d here
      inits = list()
    ),
    NNGP = list(
      code = quote({
        AD[1:N,1:(k+1)] <- calculateAD_ns(dist1_3d[1:N,1:(k+1),1:(k+1)],
                                          dist2_3d[1:N,1:(k+1),1:(k+1)],
                                          dist12_3d[1:N,1:(k+1),1:(k+1)],
                                          Sigma11[1:N], Sigma22[1:N], Sigma12[1:N],
                                          log_sigma_vec[1:N], log_tau_vec[1:N],
                                          nID[1:N,1:k], N, k, nu, d)
        z[1:N] ~ dmnorm_nngp(mu[1:N], AD[1:N,1:(k+1)], nID[1:N,1:k], N, k)
      }),
      constants_needed = c("N", "coords", "d", "dist1_3d", "dist2_3d", "dist12_3d", "nID", "k", "nu"),    ## keep N, coords, d here
      inits = list()
    ),
    SGV = list(
      code = quote({
        U[1:num_NZ,1:3] <- calculateU_ns( dist1_3d[1:N,1:(k+1),1:(k+1)], 
                                          dist2_3d[1:N,1:(k+1),1:(k+1)],
                                          dist12_3d[1:N,1:(k+1),1:(k+1)],
                                          Sigma11[1:N], Sigma22[1:N], Sigma12[1:N],
                                          log_sigma_vec[1:N], log_tau_vec[1:N], 
                                          nu, nID[1:N,1:k], cond_on_y[1:N,1:k], N, k, d )
        z[1:N] ~ dmnorm_sgv(mu[1:N], U[1:num_NZ,1:3], N, k)
      }),
      constants_needed = c("N", "coords", "d", "dist1_3d", "dist2_3d", "dist12_3d", "nID", "k", "nu", "cond_on_y", "num_NZ"),    ## keep N, coords, d here
      inits = list()
    )
  )
  }
  
  ##============================================
  ## Setup
  ##============================================
  
  if(is.null(  tau_model_list[[  tau_model]])) stop("unknown specification for tau_model")
  if(is.null(sigma_model_list[[sigma_model]])) stop("unknown specification for sigma_model")
  if(is.null(Sigma_model_list[[Sigma_model]])) stop("unknown specification for Sigma_model")
  if(is.null(   mu_model_list[[   mu_model]])) stop("unknown specification for mu_model")
  if(is.null( likelihood_list[[ likelihood]])) stop("unknown specification for likelihood")
  
  model_selections_list <- list(
    tau        = tau_model_list  [[tau_model]],
    sigma      = sigma_model_list[[sigma_model]],
    Sigma      = Sigma_model_list[[Sigma_model]],
    mu         = mu_model_list   [[mu_model]],
    likelihood = likelihood_list [[likelihood]]
  )
  
  ## code
  
  code_template <- quote({
    SIGMA_MODEL             ## Log variance
    TAU_MODEL               ## Log nugget -- gotta respect the nugget
    CAP_SIGMA_MODEL         ## Anisotropy
    MU_MODEL                ## Mean
    LIKELIHOOD_MODEL        ## Likelihood
  })
  
  code <-
    eval(substitute(substitute(
      CODE,
      list(TAU_MODEL        = model_selections_list$tau$code,
           SIGMA_MODEL      = model_selections_list$sigma$code,
           CAP_SIGMA_MODEL  = model_selections_list$Sigma$code,
           CAP_SIGMA_MODEL  = model_selections_list$Sigma$code,
           MU_MODEL         = model_selections_list$mu$code,
           LIKELIHOOD_MODEL = model_selections_list$likelihood$code)),
      list(CODE = code_template)))
  
  if(missing(data)) stop("must provide data as 'data' argument")
  N <- length(data)
  
  if(missing(coords)) stop("must provide 'coords' argument, array of spatial coordinates")
  d <- ncol(coords)
  coords <- as.matrix(coords)
  
  sd_default <- 100
  mu_default <- 0
  matern_rho_default <- 1
  matern_nu_default <- 5  
  maxDist <- 0
  for(j in 1:d){
    maxDist <- maxDist + (max(coords[,j]) - min(coords[,j]))^2
  }
  maxDist <- sqrt(maxDist) # max(dist(coords))
  
  if(N < 200){
    ones_set <- rep(1,200); zeros_set <- rep(1,200)
  } else{
    ones_set <- rep(1,N); zeros_set <- rep(1,N)
  }
  
  constants_defaults_list <- list(
    N = N,
    coords = coords,
    d = d,
    zeros = zeros_set,
    ones = ones_set,
    mu_HP1 = sd_default,                 ## standard deviation
    tau_HP1 = sd_default,                ## standard deviation/upper bound for constant nugget
    tau_HP2 = matern_nu_default,         ## approxGP smoothness
    tau_HP3 = maxDist,                   ## upper bound for approxGP range
    tau_HP4 = sd_default,                ## upper bound for approxGP sd
    sigma_HP1 = sd_default,              ## standard deviation
    sigma_HP2 = matern_nu_default,       ## approxGP smoothness
    sigma_HP3 = maxDist,                 ## upper bound for approxGP range
    sigma_HP4 = sd_default,              ## upper bound for approxGP sd
    Sigma_HP1 = rep(10,2),               ## standard deviation/upper bound
    Sigma_HP2 = rep(10,2),               ## uniform upper bound for covReg 'psi' parameters / latent approxGP smoothness
    Sigma_HP3 = rep(maxDist,2),          ## upper bound for approxGP range
    Sigma_HP4 = rep(sd_default, 2),      ## upper bound for approxGP sd
    maxAbsLogSD = 10,                    ## logSD must live between +/- maxAbsLogSD
    maxAnisoRange = maxDist,             ## maximum value for the diagonal elements of the anisotropy process
    minAnisoDet = 1e-5,                  ## lower bound for the determinant of the anisotropy process
    nu = matern_nu_default               ## Process smoothness parameter
  )
  
  ## use the isotropic model?
  useIsotropic <- (Sigma_model %in% c("constantIso", "compRegIso", "npApproxGPIso"))
  
  ## initialize constants_to_use with constants_defaults_list
  constants_to_use <- constants_defaults_list
  
  ## update constants_to_use with those arguments provided via ...
  dotdotdot <- list(...)
  ## make sure all ... arguments were provided with names
  if(length(dotdotdot) > 0 && (is.null(names(dotdotdot)) || any(names(dotdotdot) == ""))) stop("Only named arguemnts should be provided through ... argument")
  constants_to_use[names(dotdotdot)] <- dotdotdot
  
  ## add 'constants' argument to constants_to_use list:
  ## if provided, make sure 'constants' argument is a named list
  if(!missing(constants)) {
    if(length(constants) > 0 && (is.null(names(constants)) || any(names(constants) == ""))) stop("All elements in constants list argument must be named")
    constants_to_use[names(constants)] <- constants
  }
  
  ## generate and add dist1_sq, dist2_sq, and dist12 arrays to constants_to_use list
  if(likelihood == 'fullGP') {
    dist_list <- nsDist(coords = coords, isotropic = useIsotropic)
  } else { ## likelihood is NNGP, or SGV:
    if(is.null(constants_to_use$k)) stop(paste0('missing k constants argument for ', likelihood, ' likelihood'))
    mmd.seed <- sample(1e5, 1) # Set seed for reproducibility (randomness in orderCoordinatesMMD function)
    if(likelihood == 'NNGP') {
      # cat("\nOrdering the prediction locations and determining neighbors for NNGP (this may take a minute).\n")
      # Re-order the coordinates/data
      coords_mmd <- orderCoordinatesMMD(coords)
      ord <- coords_mmd$orderedIndicesNoNA
      coords <- coords[ord,]
      data <- data[ord]
      # Set neighbors and calculate distances
      nID <- determineNeighbors(coords, constants_to_use$k)
      constants_to_use$nID <- nID
      dist_list <- nsDist3d(coords = coords, nID = nID, isotropic = useIsotropic)
    }
    if(likelihood == 'SGV') {
      # cat("\nOrdering the prediction locations and determining neighbors/conditioning sets for SGV (this may take a minute).\n")
      if(is.null(sgvOrdering)){
        setupSGV <- sgvSetup(coords = coords, k = constants_to_use$k, seed = mmd.seed)
      }else{
        setupSGV <- sgvOrdering
      }
      constants_to_use$nID <- setupSGV$nID_ord
      constants_to_use$cond_on_y <- setupSGV$condition_on_y_ord
      constants_to_use$num_NZ <- setupSGV$num_NZ
      ord <- setupSGV$ord
      # Re-order the coordinates/data
      coords <- coords[ord,]
      data <- data[ord]
      dist_list <- nsDist3d(coords = coords, nID = setupSGV$nID_ord, isotropic = useIsotropic)
    }
    # Re-order any design matrices
    if(!is.null(constants_to_use$X_tau)) constants_to_use$X_tau <- constants_to_use$X_tau[ord,]
    if(!is.null(constants_to_use$X_sigma)) constants_to_use$X_sigma <- constants_to_use$X_sigma[ord,]
    if(!is.null(constants_to_use$X_Sigma)) constants_to_use$X_Sigma <- constants_to_use$X_Sigma[ord,]
    if(!is.null(constants_to_use$X_mu)) constants_to_use$X_mu <- constants_to_use$X_mu[ord,]
  }
  constants_to_use$coords <- coords
  constants_to_use[names(dist_list)] <- dist_list
  
  ## if any models use approxGP: calculate XX_knot_dist and XX_cross_dist (coords already re-ordered for NNGP/SGV)
  if( tau_model == 'approxGP' ) {
    if(is.null(constants_to_use$tau_knot_coords)) stop(paste0('missing tau_knot_coords for tau_model: approxGP'))
    constants_to_use$tau_knot_dist <- sqrt(nsDist(coords = constants_to_use$tau_knot_coords, isotropic = TRUE)$dist1_sq)
    constants_to_use$tau_cross_dist <- sqrt(nsCrossdist(Pcoords = coords, coords = constants_to_use$tau_knot_coords, isotropic = TRUE)$dist1_sq)
  }
  if( sigma_model == 'approxGP' ) {
    if(is.null(constants_to_use$sigma_knot_coords)) stop(paste0('missing sigma_knot_coords for sigma_model: approxGP'))
    constants_to_use$sigma_knot_dist <- sqrt(nsDist(coords = constants_to_use$sigma_knot_coords, isotropic = TRUE)$dist1_sq)
    constants_to_use$sigma_cross_dist <- sqrt(nsCrossdist(Pcoords = coords, coords = constants_to_use$sigma_knot_coords, isotropic = TRUE)$dist1_sq)
  }
  if( Sigma_model %in% c('npApproxGP', 'npApproxGPIso') ) {
    if(is.null(constants_to_use$Sigma_knot_coords)) stop(paste0('missing Sigma_knot_coords for Sigma_model: ', Sigma_model))
    constants_to_use$Sigma_knot_dist <- sqrt(nsDist(coords = constants_to_use$Sigma_knot_coords, isotropic = TRUE)$dist1_sq)
    constants_to_use$Sigma_cross_dist <- sqrt(nsCrossdist(Pcoords = coords, coords = constants_to_use$Sigma_knot_coords, isotropic = TRUE)$dist1_sq)
  }
  
  ## add the following (derived numbers of columns) to constants_to_use:
  ## p_tau:
  if(!is.null(constants_to_use$X_tau)) constants_to_use$p_tau <- ncol(constants_to_use$X_tau)
  if(!is.null(constants_to_use$tau_cross_dist)) constants_to_use$p_tau <- ncol(constants_to_use$tau_cross_dist)
  ## p_sigma:
  if(!is.null(constants_to_use$X_sigma)) constants_to_use$p_sigma <- ncol(constants_to_use$X_sigma)
  if(!is.null(constants_to_use$sigma_cross_dist)) constants_to_use$p_sigma <- ncol(constants_to_use$sigma_cross_dist)
  ## p_Sigma:
  if(!is.null(constants_to_use$X_Sigma)) constants_to_use$p_Sigma <- ncol(constants_to_use$X_Sigma)
  if(!is.null(constants_to_use$Sigma_cross_dist)) constants_to_use$p_Sigma <- ncol(constants_to_use$Sigma_cross_dist)
  ## p_mu:
  if(!is.null(constants_to_use$X_mu)) constants_to_use$p_mu <- ncol(constants_to_use$X_mu)
  
  ## get a vector of all the constants we need for this model
  constants_needed <- unique(unlist(lapply(model_selections_list, function(x) x$constants_needed), use.names = FALSE))
  ## check if we're missing any constants we need, and throw an error if any are missing
  constants_missing <- setdiff(constants_needed, names(constants_to_use))
  if(length(constants_missing) > 0) {
    stop(paste0("Missing values for the following model constants: ",
                paste0(constants_missing, collapse = ", "),
                ".\nThese values should be provided as named arguments, or named elements in the constants list argument"))
  }
  
  ## generate the constants list
  constants <- constants_to_use[constants_needed]
  
  ## append the mmd.seed and ord for SGV/NNGP
  if(likelihood != 'fullGP') {
    constants$mmd.seed <- mmd.seed
    constants$ord <- ord
  }
  
  ## ensure Sigma_HPX parameters are vectors of length 2
  if(!is.null(constants$Sigma_HP1)){
    if(length(constants$Sigma_HP1) == 1) constants$Sigma_HP1 <- rep(constants$Sigma_HP1, 2)
  }
  if(!is.null(constants$Sigma_HP2)){
    if(length(constants$Sigma_HP2) == 1) constants$Sigma_HP2 <- rep(constants$Sigma_HP2, 2)
  }
  if(!is.null(constants$Sigma_HP3)){
    if(length(constants$Sigma_HP3) == 1) constants$Sigma_HP3 <- rep(constants$Sigma_HP3, 2)
  }
  if(!is.null(constants$Sigma_HP4)){
    if(length(constants$Sigma_HP4) == 1) constants$Sigma_HP41 <- rep(constants$Sigma_HP4, 2)
  }
  
  ## gather constraints_needed data
  constraints_needed <- unique(unlist(lapply(model_selections_list, function(x) x$constraints_needed), use.names = FALSE))
  constraints_data <- as.list(rep(1, length(constraints_needed)))
  names(constraints_data) <- constraints_needed
  
  print("Imposed constraints:")
  print(constraints_needed)
  ## data
  data <- c(list(z = data), constraints_data)
  
  ## inits
  inits_uneval <- do.call("c", unname(lapply(model_selections_list, function(x) x$inits)))
  inits <- lapply(inits_uneval, function(x) eval(x, envir = constants))
  
  # if(returnModelComponents) return(list(code=code, constants=constants, data=data, inits=inits))
  
  ## generate the "name" for the nimble model object, containing which submodels were used
  thisName <- paste0(
    'tau='       , tau_model  , '_',
    'sigma='     , sigma_model, '_',
    'Sigma='     , Sigma_model, '_',
    'mu='        , mu_model   , '_',
    'likelihood=', likelihood
  )
  
  ## register custom NNGP or SGV distributions (if necessary),
  ## importantly, using mixedSizes = TRUE to avoid warnings
  if(likelihood == 'NNGP') {
    registerDistributions(list(
      dmnorm_nngp = list(
        BUGSdist = 'dmnorm_nngp(mean, AD, nID, N, k)',
        types = c('value = double(1)', 'mean = double(1)', 'AD = double(2)', 'nID = double(2)', 'N = double()', 'k = double()'),
        mixedSizes = TRUE)
    ), verbose = FALSE)
  }
  if(likelihood == 'SGV') {
    registerDistributions(list(
      dmnorm_sgv = list(
        BUGSdist = 'dmnorm_sgv(mean, U, N, k)',
        types = c('value = double(1)', 'mean = double(1)', 'U = double(2)', 'N = double()', 'k = double()'),
        mixedSizes = TRUE)
    ), verbose = FALSE)
  }
  
  ## NIMBLE model object
  Rmodel <- nimbleModel(code, constants, data, inits, name = thisName)
  lp <- Rmodel$getLogProb()
  if(is(lp, 'try-error') || is.nan(lp) || is.na(lp) || abs(lp) == Inf) stop('model not properly initialized')
  
  ## store 'constants' list into Rmodel$isDataEnv
  Rmodel$isDataEnv$.BayesNSGP_constants_list <- constants
  
  ## using the nsgpModel() argument monitorAllSampledNodes,
  ## set nimble package option: MCMCmonitorAllSampledNodes,
  ## so that latent process values are monitored by default, for use in predicition
  nimbleOptions(MCMCmonitorAllSampledNodes = monitorAllSampledNodes)
  
  return(Rmodel)
}
