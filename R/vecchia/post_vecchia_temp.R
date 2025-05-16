############################################################################
# POSTERIOR SAMPLING WITH VECCHIA MODEL
############################################################################

#----------------------------------------------------------------

## Load function to create Vecchia model
source("setupVecchiaTemp.R")

#----------------------------------------------------------------

# Define posterior parameters
nBatch=50
Ngd=256
n_samples_max=50000
tau2obs=0.05**2

path_files="./vecchia_temp/"

#--------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if(sum(is.na(args[1:3]))>0){
  stop("Three numerical argumuments are expected for the first three arguements of the R script")
}
id_sim=as.numeric(args[1]) #  Id of the full simulation
id_mask=as.numeric(args[2]) # Id of the mask
kVecchia=as.numeric(args[3])
subsamp=as.numeric(args[4])

print(args)

id_run=as.numeric(args[5]) # Id of the call
nArrayRuns=as.numeric(args[6])+1 # Number of parallel runs

export_folder=args[7]

#--------------------------------------------------------------
## File paths


## Id file and export folder
if(is.na(subsamp)){
  id_file=paste0("run_",id_sim,id_mask,"_v",kVecchia)
}else{
  id_file=paste0("run_",id_sim,id_mask,"_v",kVecchia,"_s",subsamp)
}
if(is.na(export_folder)){
  export_folder=paste0(path_files,id_file,"/")
}
id_file=paste0(id_file,"_")

#--------------------------------------------------------------

cat("MCMC data: Data Id=",id_sim,"/ Mask Id=",id_mask,"/ Vecchia Approx=",kVecchia,
    "/ Subsamp=",subsamp,"\n")

#--------------------------------------------------------------

## Export file name
filename=paste0(path_files,id_file,id_run,"_post.h5")

#--------------------------------------------------------------

## LOAD SAMPLES

## Result files
pattern <- paste0("^",".*", "\\.csv$")
files <- list.files(path = export_folder, pattern = pattern, full.names = F)

## Extract parameters
param_list=as.matrix(do.call(rbind,lapply(files, function(l){read.csv(paste0(export_folder,l))})))
vnames=c("Sigma_coef1", "Sigma_coef2", paste0("Sigma_coef3[",1:36,"]"))
nparams_tot=min(nrow(param_list),n_samples_max)
ipp=floor(seq(from=0,to=nparams_tot,length.out=nArrayRuns+1))
param_list=matrix(param_list[(ipp[id_run+1]+1):(ipp[id_run+2]),],ncol=ncol(param_list))
colnames(param_list)=vnames
cat("Computing samples",(ipp[id_run+1]+1),"to",(ipp[id_run+2]),"and saving results in file",filename,"\n")


## Number of posterior samples
npost=nrow(param_list)
nparams=ncol(param_list)
if(npost<nBatch){
  nBatch=npost
}
nruns=ceiling(npost / nBatch)

cat("Found",npost,"posterior samples...\n")


#--------------------------------------------------------------

cat("Creating Vecchia Model...\n")

res=createVecchiaModel(path_files,id_file,id_sim,id_mask,kVecchia,tau2obs,subsamp,constInit=TRUE)
Rmodel=res$model
Yobs=res$data
obsCoord=res$obsCoord
nodeMat=res$allCoord
sel_mask=res$indObs
z_sim=res$image
param_sim=res$param_sim
mask=res$mask
knot_blocks=res$knot_blocks
xknots=res$xknots
ordObs=res$ord
mObs=res$mObs
sdObs=res$sdObs
rm(res)

#--------------------------------------------------------------
#--------------------------------------------------------------

## Prediction locations
sel_pred=setdiff(1:nrow(nodeMat),sel_mask)
Xpred=nodeMat[sel_pred,]
matPredSplines=setSplineCovarMatrix(Xpred,xknots)
constants=list()
constants[["PX_Sigma"]]=matPredSplines

orderingFile=paste0(path_files,id_file,"pred_ordering.rds")
if(file.exists(orderingFile)){
  message("\n Found existing ordering of prediction locations.\n")
  sgvOrdering=readRDS(orderingFile)
}else{
  message("\nOrdering the prediction locations and determining neighbors/conditioning sets for SGV (this may take a minute).\n")
  sgvOrdering <- sgvSetup(coords = obsCoord[ordObs,], coords_pred = Xpred, k = kVecchia, 
                          seed=sample(1e5,1),order_coords = FALSE)
  saveRDS(sgvOrdering,orderingFile)
}


## Prediction function
.computeSim<-function(i){

  ## Select samples
  spred=matrix(param_list[i,],ncol = nparams)
  colnames(spred)=vnames
  
  ## predict at target locations
  uu=nsgpPredict(Rmodel,samples = spred,coords.predict = Xpred,
                 sgvOrdering=sgvOrdering,
                 constants = constants)

  ## Build full image
  zpred=matrix(0,nrow=nrow(nodeMat),ncol=length(i))
  zpred[sel_mask,]=t(uu$obs)
  zpred[sel_pred,]=t(uu$pred)
  
  zpred=mObs+sdObs*zpred
  
  # ## Build full image
  # zk=matrix(0,nrow=nrow(nodeMat),ncol=length(i))
  # zk[sel_mask,]=t(uu$krig_obs)
  # zk[sel_pred,]=t(uu$krig_pred)

  return(zpred)
}

if (file.exists(filename)) {
  file.remove(filename)
}
h5createFile(filename)
h5createDataset(filename, "data", rev(c(1,2,Ngd,Ngd)), maxdims = rev(c(1,2,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "post", rev(c(nBatch,1,Ngd,Ngd)), maxdims = rev(c(npost,1,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "post_params", rev(c(nBatch,nparams)),maxdims = rev(c(npost,nparams)),
                native=TRUE)

h5write(z_sim, file=filename, name="data", index=list(1:Ngd,1:Ngd,1,1))
h5write(mask, file=filename, name="data", index=list(1:Ngd,1:Ngd,2,1))

cnt=1
for(k in 1:nruns){
  print(paste0("Configuration ",k,"/",nruns))

  iruns=seq(from=(k-1)*nBatch+1, to=min(npost,k*nBatch))

  h5set_extent(filename, "post",c(Ngd,Ngd,1,iruns[length(iruns)]))
  h5set_extent(filename, "post_params",c(nparams,iruns[length(iruns)]))

  res_sim= .computeSim(iruns)

  # Store
  res_sim=array(res_sim,dim=c(Ngd,Ngd,1,ncol(res_sim)))
  h5write(res_sim,file=filename, name="post", index=list(1:Ngd,1:Ngd,1,iruns))
  h5write(t(param_list[iruns,]), file=filename, name="post_params", index=list(1:nparams,iruns))

}
h5closeAll()

