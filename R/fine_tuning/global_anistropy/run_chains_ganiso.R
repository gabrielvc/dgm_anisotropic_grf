############################################################################
# simulation test on a square
############################################################################

library(NSGP)
library(LaplacesDemon)
library(Matrix)
library(rhdf5)
library(fields)
library(parallel)
#----------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

# Define MCMC parameters
nChains=20
ntot_iter=25000
nRuns=as.numeric(args[5])
id_call=as.integer(Sys.time())

#--------------------------------------------------------------
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

id_call=args[6]


cat("MCMC data: Mask size=",mask_level,"/ Sim Id=",id_sim,"/ Mask Id=",id_mask,"/ Unif=",use_unif,"\n")
#--------------------------------------------------------------

tau2obs=0.01**2

L=1
Ngd=256
xseq=seq(0,L,length.out=Ngd)
nodeMat=as.matrix(expand.grid(xseq,xseq))

plt=FALSE

#--------------------------------------------------------------
#--------------------------------------------------------------


## Load data 
fn_sim=paste0(path_files,"data_ganiso_mcmc.h5")
dat_sim=h5dump(fn_sim)
z_sim=dat_sim$data[,,1,id_sim]
param_sim=dat_sim$params[,id_sim]

## Load Mask
fn_mask=paste0(path_files,"data_mcmc_mask.h5")
dat_mask=h5dump(fn_mask)
if(use_clust_mask){
  mask=c(dat_mask$clust[, , mask_level,id_mask])
  sel_mask=which(mask!=0)
  obs_noise=c(dat_mask$noise_clust[, ,mask_level,id_mask])[sel_mask]
}else{
  mask=c(dat_mask$unif[, , mask_level,id_mask])
  sel_mask=which(mask!=0)
  obs_noise=c(dat_mask$noise_unif[, ,mask_level,id_mask])[sel_mask]
}
rm(dat_mask,dat_sim)

Yobs=c(z_sim)[sel_mask] + obs_noise
obsCoord=nodeMat[sel_mask,]
nobs=length(sel_mask)
cat("Number of observations : ",nobs,"\n")

if(plt){
  image.plot(matrix(z_sim,Ngd))
  points(nodeMat[sel_mask,],cex=2*obs_noise/max(abs(obs_noise)))
}

#--------------------------------------------------------------

id_file=paste0("run_",mask_level,id_sim,id_mask)

export_folder=paste0(path_files,id_file,"/")
if(!dir.exists(export_folder)){
  dir.create(export_folder)
}

id_file=paste0(id_file,"_")


#--------------------------------------------------------------


rmcmc<-function(x){
  tt=runif(1,0,pi)
  # Varaible scale
  amin=0.05; amax=0.3
  a=runif(1,amin,amax)
  ## Anistropy ratio
  param_rho_scale=c(1,runif(1,0.1,1))
  ## Regularity
  nu=2
  
  theta_init=c(a,nu,tt,param_rho_scale[2])
  run=RW_MCMC(theta_init,Yobs,obsCoord,tau2obs,
              nbIter = ntot_iter,
              Thinning = 50,
              stdVec = 0.05*c(1,0,1,1))
  colnames(run)=c("a","nu","theta","rho")
  return(run)
}

#--------------------------------------------------------------
#--------------------------------------------------------------

# Run MCMC in batches
for (i in 1:nRuns) {
  
  cat("\n Run # ", i, "/", nRuns,"...\n")
  
  fit <- mclapply(1:nChains, rmcmc, mc.cores = nChains)
  
  # Save checkpoint
  saveRDS(fit, paste0(export_folder,id_file,"rwm_", id_call, "_" ,i, ".rds"))
  
  # Update initial values for next batch
  final_values <- as.matrix(do.call(rbind,lapply(fit,function(x){x[nrow(x),]})))
  write.csv(final_values,paste0(export_folder,id_file,"rwm_", id_call, "_" ,i, ".csv"),row.names = F)
}

