


#' Random field simulation
#'
#' Generate samples of a Gaussian random field specified by a power spectral density and FEM matrices.
#'
#'@param psd Function defining the power spectral density of the field.
#'@param FEMatList List of FEM matrices obtained from the \code{matFEM} function.
#'@param nbsimu Number of samples to generate (Default=1).
#'@param seed Seed for random number generator (Default=NULL).
#'@param WN Matrix whose columns contains vectors of Gaussian white noise to be used to generate the samples. If set to NULL (Default), this matrices is generated on the fly. If set, \code{nbsimu} and \code{seed} are ignored. Note: The number of rows should be equal to the size of the FEM matrices.
#'@param tolCoefs Tolerance to discard Chebyshev coefficients (Default=10^{-4}). The coefficients below tolCoefs*maximum coeffients are discarded.
#'@param verbose Boolean. Print messages or not.
#'
#'@return A matrix containing the generated samples.
#'
simPSD<-function(psd,FEMatList,nbsimu=1,seed=NULL,WN=NULL,tolCoefs=10**(-4),verbose=TRUE){

  ## Unpack matrices
  S=FEMatList$Shift
  Cmd=FEMatList$Scale
  lmin=FEMatList$Eig[1]
  lmax=FEMatList$Eig[2]

  N=nrow(S)

  ## Approximation Chebyshev
  OC=500
  idm=NULL
  while(length(idm)<=0){
    cf=coefsChebApprox(psd,OC,lmin,lmax)
    idm=which(abs(cf)<(max(abs(cf))*tolCoefs))
    OC=2*OC
  }
  cf=cf[-idm]
  if(verbose){
    print(paste0("Number of coefficients used in Chebyshev approximation = ",length(cf)))
  }

  ## Simulation
  if(!is.null(seed)){
    set.seed(seed)
  }
  z=NULL
  if(is.null(WN)){
    if(nbsimu>0){
      z=Cmd%*%prodChebDecompVec(S,cf,lmin,lmax,matrix(rnorm(N*nbsimu),ncol = nbsimu))
    }
  }else{
    if(nrow(as.matrix(WN))==dim(S)[1]){
      z=Cmd%*%prodChebDecompVec(S,cf,lmin,lmax,as.matrix(WN))
    }else{
      stop("The number of rows in the white noise matrix should be equal to the number of rows of the FEM matrices.")
    }
  }


  return(z)

}




#'
#'@rdname simPSD
#'
simPSDChol<-function(psd,FEMatList,nbsimu=1,seed=NULL,WN=NULL,tolCoefs=10**(-4),verbose=TRUE){
  
  ## Unpack matrices
  R=FEMatList$Stiff
  C=FEMatList$Mass
  lmin=FEMatList$Eig[1]
  lmax=FEMatList$Eig[2]
  
  s=summary(R)
  s=sparseMatrix(i=s[,1],j=s[,2],x=abs(s[,3]))
  lmax=min(max(rowSums(s)),max(colSums(s)))
  rm(s)
  gc()
  lmax=lmax/min(diag(C)/2)
  
  
  N=nrow(R)
  
  ## Approximation Chebyshev
  OC=500
  idm=NULL
  while(length(idm)<=0){
    cf=coefsChebApprox(psd,OC,lmin,lmax)
    idm=which(abs(cf)<(max(abs(cf))*tolCoefs))
    OC=2*OC
  }
  cf=cf[-idm]
  if(verbose){
    print(paste0("Number of coefficients used in Chebyshev approximation = ",length(cf)))
  }
  
  ## Simulation
  set.seed(seed)
  z=NULL
  if(is.null(WN)){
    if(nbsimu>0){
      return(simChebChol(C,R,cf,lmin,lmax,matrix(rnorm(N*nbsimu),ncol = nbsimu)))
    }
  }else{
    if(nrow(as.matrix(WN))==N){
      return(simChebChol(C,R,cf,lmin,lmax,as.matrix(WN)))
    }else{
      stop("The number of rows in the white noise matrix should be equal to the number of rows of the FEM matrices.")
    }
  }
  
}
