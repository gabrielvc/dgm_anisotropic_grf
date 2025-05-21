############################################################################
# simulations on a square
############################################################################

library(SFEMsim)
library(RTriangle)
library(Matrix)
library(fields)
library(rhdf5)
library(spatstat.random)

#--------------------------------------------------------------
#--------------------------------------------------------------

## Parameters of the simulation domain
L=1 # Domain size
Ngd=256 # Number of grid points
Ngd_ext=32 # Number of additional grid points for boundary effects
nknots_grid=6 # Number of spline knots (for non stationary parameters)

##  Parameters of sample simulations
nex_samples=10 # Number of parameter configurations
nsim_samples=1 # Number of simulations per parameter configuration


## Parameters of mask generations
mask_sizes=c(50,100,300,600,1000) # Mask sizes
lambda_parent=10 # Mean number of clusters
child_radius=0.1 # Radius of children
tau_noise=0.01 # Noise level (std)
nex=10 # Number of masks generated per size


## Export folder
export_folder="./"

## Prefix of saved datasets
filename0="data_mcmc"

## Seed
set.seed(42)


#--------------------------------------------------------------
#--------------------------------------------------------------

gen_sim_poisClus<-function(lambda_tot,lambda_parent,child_radius,expand=0.5,
                           nGrid=NULL){
  ## Mean number of points per cluster
  lambda_child=lambda_tot/lambda_parent
  
  ## Function to generate clusters
  nclust <- function(x0, y0, radius, lambda) {
    n=rpois(1,lambda)
    return(runifdisc(n, radius, centre=c(x0, y0)))
  }
  
  sim_res=rPoissonCluster(lambda_parent, 
                          expand=expand, 
                          rcluster=nclust, radius=child_radius, lambda=lambda_child,
                          saveparents = TRUE,
                          nsim=1)
  sim_res=cbind(sim_res$x,sim_res$y)
  
  if(!is.null(nGrid)){
    projGrid=expand.grid(x=seq(from=0,to=1,length.out=nGrid[1]),y=seq(from=0,to=1,length.out=nGrid[2]))
    indProj=apply(sim_res, 1, function(pt){which.min((projGrid[,1]-pt[1])**2+(projGrid[,2]-pt[2])**2)})
    indProj=unique(indProj)
    sim_res=projGrid[indProj,]
  }else{
    indProj=NULL
  }
  
  return(list(sim=sim_res,ind=indProj))
}


#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------

### SIMULATION OF MASKS

nsizemask=length(mask_sizes)

filename=paste0(export_folder,filename0,"_mask",".h5")
if (file.exists(filename)) {
  #Delete file if it exists
  file.remove(filename)
}
h5createFile(filename)
h5createDataset(filename, "unif", rev(c(1,nsizemask,Ngd,Ngd)), maxdims = rev(c(nex,nsizemask,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "noise_unif", rev(c(1,nsizemask,Ngd,Ngd)), maxdims = rev(c(nex,nsizemask,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "clust", rev(c(1,nsizemask,Ngd,Ngd)), maxdims = rev(c(nex,nsizemask,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "noise_clust", rev(c(1,nsizemask,Ngd,Ngd)), maxdims = rev(c(nex,nsizemask,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
cnt=1
for(k in 1:nex){
  print(paste0("Configuration ",k,"/",nex))
  
  ## Generate masks
  z_mask_unif=z_mask_clust=noise_mask_unif=noise_mask_clust=matrix(0,nrow = Ngd*Ngd,ncol = nsizemask)
  for(l in 1:nsizemask){
    indProj=sample(1:nrow(z_mask_unif),mask_sizes[l],replace = FALSE)
    z_mask_unif[indProj,l]=1
    noise_mask_unif[indProj,l]=tau_noise*rnorm(length(indProj))
    
    indProj=gen_sim_poisClus(mask_sizes[l],lambda_parent,child_radius,nGrid = c(Ngd,Ngd))$ind
    z_mask_clust[indProj,l]=1
    noise_mask_clust[indProj,l]=tau_noise*rnorm(length(indProj))
    
  }
  
  # Store
  for (i in 1:nsizemask) {
    h5write(matrix(z_mask_unif[,i],Ngd,Ngd), file=filename, name="unif", index=list(1:Ngd,1:Ngd,i,cnt))
    h5write(matrix(z_mask_clust[,i],Ngd,Ngd), file=filename, name="clust", index=list(1:Ngd,1:Ngd,i,cnt))
    h5write(matrix(noise_mask_unif[,i],Ngd,Ngd), file=filename, name="noise_unif", index=list(1:Ngd,1:Ngd,i,cnt))
    h5write(matrix(noise_mask_clust[,i],Ngd,Ngd), file=filename, name="noise_clust", index=list(1:Ngd,1:Ngd,i,cnt))
  }
  cnt=cnt+1
  
  if(k<nex){
    h5set_extent(filename, "unif",c(Ngd,Ngd,nsizemask,cnt))
    h5set_extent(filename, "clust",c(Ngd,Ngd,nsizemask,cnt))
    h5set_extent(filename, "noise_unif",c(Ngd,Ngd,nsizemask,cnt))
    h5set_extent(filename, "noise_clust",c(Ngd,Ngd,nsizemask,cnt))
  }
  
}

h5closeAll()


#--------------------------------------------------------------

