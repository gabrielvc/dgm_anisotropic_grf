############################################################################
# CREATE VECCHIA MODEL FROM BAYESNSGP PACKAGE
############################################################################

library(NSGP)
library(Matrix)
library(rhdf5)
library(BayesNSGP)

#------------------------------------------------------------------

setSplineCovarMatrix<-function(predCoord,xknots){
  
  matPredSplines=matCovarTPS(predCoord,xknots)
  
  M_TPS=getTPSvalMat(xknots)
  
  PermMat=matrix(0,42,42)
  PermMat[cbind(1:42,c(37:42,1:36))]=1
  
  ExtMat=matrix(0,42,36)
  diag(ExtMat[1:36,1:36])=1
  
  matCovar=matPredSplines %*% PermMat%*% M_TPS %*% ExtMat
  
  return(matCovar)
}

createVecchiaModel<-function(path_files,id_file,id_sim,id_mask,kVecchia,tau2obs,subsamp,
                             constInit=F,onlyOrd=F,sgvOrdering=NULL){
  
  L=1
  Ngd=256
  xseq=seq(0,L,length.out=Ngd)
  nodeMat=as.matrix(expand.grid(xseq,xseq))
  
  ## Regular grid of points in the domain
  Ngd_ext=32
  Ngd_tot=Ngd+Ngd_ext*2
  h=xseq[2]-xseq[1]
  xseq_ext=c(-(Ngd_ext:1)*h,xseq,(1:Ngd_ext)*h+L)
  
  ## Knots
  nknots_grid=6
  xspl=seq(from=min(xseq_ext),to=max(xseq_ext),length.out=nknots_grid)
  xknots=as.matrix(expand.grid(xspl,xspl))
  nknots=nrow(xknots)
  
  knot_blocks=list(which((xknots[,1]<0.5)*(xknots[,2]<0.5)==1),
                   which((xknots[,1]<0.5)*(xknots[,2]>=0.5)==1),
                   which((xknots[,1]>=0.5)*(xknots[,2]<0.5)==1),
                   which((xknots[,1]>=0.5)*(xknots[,2]>=0.5)==1))
  
  
  #--------------------------------------------------------------
  #--------------------------------------------------------------
  
  ## Load data 
  fn_sim=paste0(path_files,"data_temp_anomb.h5")
  dat_sim=h5dump(fn_sim)
  z_sim=dat_sim$data[,,id_sim]
  if(!is.na(subsamp)){
    if(subsamp==300){
      mask=dat_sim$mask300[,,id_mask]
    }else if(subsamp==600){
      mask=dat_sim$mask600[,,id_mask]
    }else if(subsamp==1000){
      mask=dat_sim$mask1000[,,id_mask]
    }else if(subsamp==2000){
      mask=dat_sim$mask2000[,,id_mask]
    }else{
      stop("Subsample size should be 300, 600 or 1000")
    }
  }else{
    mask=dat_sim$mask[,,id_mask]
  }
  sel_mask=which(c(mask)!=0)
  
  rm(dat_sim)
  
  Yall=c(z_sim)
  mObs=mean(Yall[sel_mask])
  sdObs=sd(Yall[sel_mask])
  Yobs=(Yall[sel_mask]-mObs)/sdObs
  
  obsCoord=nodeMat[sel_mask,]
  nobs=length(sel_mask)
  cat("Number of observations : ",nobs,"\n")
  
  ## Spline covariate matrix
  matObsSplines=setSplineCovarMatrix(obsCoord,xknots)
  
  #--------------------------------------------------------------
  ## Setup Vecchia
  constants = list(k=kVecchia,nu = 2,
                   tau_HP1=tau2obs**0.5,
                   sigma_HP1=1,
                   coords=obsCoord,
                   X_Sigma=matObsSplines
  )
  
  ## Ordering...
  if(is.null(sgvOrdering)){
    orderingFile=paste0(path_files,id_file,"obs_ordering.rds")
    if(file.exists(orderingFile)){
      sgvOrdering=readRDS(orderingFile)
    }else{
      message("\nOrdering the observation locations and determining neighbors/conditioning sets for SGV (this may take a minute).\n")
      sgvOrdering <- sgvSetup(coords = obsCoord, k = kVecchia, 
                              seed=sample(1e5,1))
      saveRDS(sgvOrdering,orderingFile)
    }
  }

  if(onlyOrd){
    return(sgvOrdering)
  }
  
  Rmodel <- nsgpModelSimple(likelihood = "SGV", 
                            mu_model = "zero",
                            Sigma_model = "custom",
                            tau_model="fixed",
                            sigma_model = "fixed",
                            constants = constants, 
                            coords = constants$coords,
                            data = Yobs,
                            constInit=constInit,
                            sgvOrdering=sgvOrdering)
  
  return(list(
    model=Rmodel,
    data=Yobs,
    obsCoord=obsCoord,
    allCoord=nodeMat,
    indObs=sel_mask,
    image=z_sim,
    param_sim=NULL,
    mask=mask,
    obs_noise_all=NULL,
    mObs=mObs,
    sdObs=sdObs,
    constants=constants,
    knot_blocks=knot_blocks,
    xknots = xknots,
    ord=Rmodel$getConstants()$ord
  ))
  
}



