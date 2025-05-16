#--------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if(sum(is.na(args[1:3]))>0){
  stop("Three numerical argumuments are expected")
}
mask_level=as.numeric(args[1])  # Mask sizes = c(50,100,300,1000) 
id_sim=as.numeric(args[2]) #  Id of the full simulation
id_mask=as.numeric(args[3]) # Id of the mask


# Clust or not
use_unif=!(args[4]=="clust")
if(use_unif){
  path_files="./mcmc_stat_unif/"
  use_clust_mask=F
}else{
  path_files="./mcmc_stat_clust/"
  use_clust_mask=T
}

id_run=args[5]


data_files=path_files
export_files=path_files
path_files=paste0(path_files,"run_",mask_level,id_sim,id_mask,"/")

id_file=paste0("run_",mask_level,id_sim,id_mask,"_")

filename=paste0(export_files,id_file,id_run,"_post.h5")
if (file.exists(filename)) {
  #Delete file if it exists
  file.remove(filename)
}


#--------------------------------------------------------------


# Define MCMC parameters
mc.cores=16
nBatch=16*10 ## Number of samples computed in parallel
plt=F

#--------------------------------------------------------------
tau2obs=0.01**2
#--------------------------------------------------------------


#--------------------------------------------------------------
#--------------------------------------------------------------

library(NSGP)
library(LaplacesDemon)
library(Matrix)
library(rhdf5)
library(fields)
library(RTriangle)
# source("sim_cond.R")

#--------------------------------------------------------------

#----------------------------------------------------

gen_sim_ganiso<-function(theta,nodeMat,triMat,triMetricMat,nsim=1, ind_sel=NULL){
  
  a=theta[1]
  nu=theta[2]
  tt=theta[3]
  pscale=theta[4]
  
  # Anisotropy directions
  param_f=(matrix(c(cos(-tt),sin(-tt),-sin(-tt),cos(-tt)),nrow=2)%*%t(xknots))[1,]
  
  
  ## Anistropy ratio
  param_rho_scale=c(1,pscale)
  
  ## Store parameters
  params=c(param_f,param_rho_scale,tt,a,nu)
  
  #---------------------------------------
  
  triMetricMat=triMetricCompute2d(nodeMat,triMat,as.matrix(xknots),param_f,
                                  param_rho_scale[1],param_rho_scale[2],m=3,p=2,lambda=0)
  
  ############################################################################
  
  #### FEM MATRICES
  
  FEMatList=matFEM2d(nodeMat,triMat,triMetricMat)
  
  ############################################################################
  
  ### Matern PARAMETERS
  phi=a/sqrt(8*nu)
  
  ## PSD
  fQ<-function(x){
    (1/(4*pi*(phi**2)*nu))*(1+(phi**2)*x)**(nu+1)
  }
  psd_Matern<-function(x){
    1/fQ(x)**0.5
  }
  
  ## Simulation
  z_sim=simPSD(psd_Matern,FEMatList,nbsimu = nsim)
  if(!is.null(ind_sel)){
    z_sim=z_sim[ind_sel,]
  }
  
  return(z_sim)
  
}


gen_sim_cond<-function(Yobs,id_obs,theta,nodeMat,triMat,triMetricMat,nsim=1, tau2=0, ind_sel=NULL){
  
  a=theta[1]
  nu=theta[2]
  
  ## Compute kriging
  if(!is.null(ind_sel)){
    obsCoord=nodeMat[ind_sel,][id_obs, ]
    Zk=KrigGaniso(nodeMat[ind_sel,],theta,Yobs,obsCoord,tau2)
  }else{
    obsCoord=nodeMat[id_obs, ]
    Zk=KrigGaniso(nodeMat,theta,Yobs,obsCoord,tau2)
  }
  
  ## Compute Non conditional simulations
  Znc=as.matrix(gen_sim_ganiso(theta,nodeMat,triMat,triMetricMat,nsim, ind_sel))
  
  ## Compute Kriging of simulations
  Zobs=as.matrix(Znc[id_obs,])+(tau2**0.5)*matrix(rnorm(prod(dim(as.matrix(Znc[id_obs,])))),ncol=ncol(Znc))
  if(!is.null(ind_sel)){
    Znck=apply(Zobs,2,function(Z){KrigGaniso(nodeMat[ind_sel,],theta,Z,obsCoord,tau2)})
  }else{
    Znck=apply(Zobs,2,function(Z){KrigGaniso(nodeMat,theta,Z,obsCoord,tau2)})
  }
  
  return(Zk+Znc-Znck)
  
}

#--------------------------------------------

## Result files
pattern <- paste0("^", id_file, ".*", "\\.csv$")
files <- list.files(path = path_files, pattern = pattern, full.names = F)

## Extract parameters
param_list=as.matrix(do.call(rbind,lapply(files, function(l){read.csv(paste0(path_files,l))})))

## Number of posterior samples
npost=nrow(param_list)
nparams=ncol(param_list)



#--------------------------------------------------------------


#### SIMULATION DOMAIN : SQUARE

## Domain size
L=1

## Regular grid of points in the domain
Ngd=256
Ngd_ext=32
Ngd_tot=Ngd+Ngd_ext*2
xseq=seq(0,L,length.out=Ngd)
h=xseq[2]-xseq[1]
xseq=c(-(Ngd_ext:1)*h,xseq,(1:Ngd_ext)*h+L)
nodeMat=L*as.matrix(expand.grid(xseq,xseq))
ind_sel=which((nodeMat[,1]>=0)*(nodeMat[,1]<=L)*(nodeMat[,2]>=0)*(nodeMat[,2]<=L)==1)


#----------------------------------

## Knots
nknots_grid=6
xspl=seq(from=min(nodeMat),to=max(nodeMat),length.out=nknots_grid)
xknots=as.matrix(expand.grid(xspl,xspl))
nknots=nrow(xknots)

#----------------------------------

## Delaunay triangulation
res=RTriangle::triangulate(pslg(nodeMat),S=0)
nodeMat=res$P
triMat=res$T
triMetricMat=triMetricPull2d(nodeMat,triMat)
rm(res)

## Final number of nodes
N=nrow(nodeMat)
Nf=Ngd*Ngd

#--------------------------------------------------------------

## Load data 
fn_sim=paste0(data_files,"data_ganiso_mcmc.h5")
dat_sim=h5dump(fn_sim)
z_sim=dat_sim$data[,,1,id_sim]
param_sim=dat_sim$params[,id_sim]

## Load Mask
fn_mask=paste0(data_files,"data_mcmc_mask.h5")
dat_mask=h5dump(fn_mask)
if(use_clust_mask){
  mask=(dat_mask$clust[, , mask_level,id_mask])
  sel_mask=which(c(mask)!=0)
  obs_noise_all=dat_mask$noise_clust[, ,mask_level,id_mask]
  obs_noise=c(obs_noise_all)[sel_mask]
}else{
  mask=(dat_mask$unif[, , mask_level,id_mask])
  sel_mask=which(c(mask)!=0)
  obs_noise_all=dat_mask$noise_unif[, ,mask_level,id_mask]
  obs_noise=c(obs_noise_all)[sel_mask]
}
rm(dat_mask,dat_sim)

Yobs=c(z_sim)[sel_mask] + obs_noise
obsCoord=nodeMat[ind_sel,][sel_mask,]
id_obs=sel_mask
nobs=length(sel_mask)
cat("Number of observations : ",nobs,"\n")

if(plt){
  image.plot(matrix(z_sim,Ngd))
  points(nodeMat[ind_sel,][sel_mask,],cex=2*obs_noise/max(abs(obs_noise)))
}

#--------------------------------------------------------------

nruns=ceiling(npost / nBatch)


.computeSim<-function(i){
  theta=param_list[i,]
  return(gen_sim_cond(Yobs,id_obs,theta,nodeMat,triMat,triMetricMat,
                      nsim=1, tau2=tau2obs, ind_sel=ind_sel)[,1])
}

h5createFile(filename)
h5createDataset(filename, "data", rev(c(1,3,Ngd,Ngd)), maxdims = rev(c(1,3,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "params", rev(c(1,nparams)),maxdims = rev(c(1,nparams)),
                native=TRUE)
h5createDataset(filename, "post", rev(c(nBatch,1,Ngd,Ngd)), maxdims = rev(c(npost,1,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "post_params", rev(c(nBatch,nparams)),maxdims = rev(c(npost,nparams)),
                native=TRUE)

h5write(z_sim, file=filename, name="data", index=list(1:Ngd,1:Ngd,1,1))
h5write(mask, file=filename, name="data", index=list(1:Ngd,1:Ngd,2,1))
h5write(obs_noise_all, file=filename, name="data", index=list(1:Ngd,1:Ngd,3,1))
nnp=length(param_sim)
h5write(param_sim[c(nnp-1,nnp,nnp-3,nnp-2)], file=filename, name="params", index=list(1:nparams,1))

cnt=1
for(k in 1:nruns){
  print(paste0("Configuration ",k,"/",nruns))
  
  iruns=seq(from=(k-1)*nBatch+1, to=min(npost,k*nBatch))
  
  h5set_extent(filename, "post",c(Ngd,Ngd,1,iruns[length(iruns)]))
  h5set_extent(filename, "post_params",c(nparams,iruns[length(iruns)]))
  
  res_sim=(mcsapply(iruns, .computeSim,mc.cores))
  
  # Store
  res_sim=array(res_sim,dim=c(Ngd,Ngd,1,ncol(res_sim)))
  h5write(res_sim,file=filename, name="post", index=list(1:Ngd,1:Ngd,1,iruns))
  h5write(t(param_list[iruns,]), file=filename, name="post_params", index=list(1:nparams,iruns))
  
}
h5closeAll()



