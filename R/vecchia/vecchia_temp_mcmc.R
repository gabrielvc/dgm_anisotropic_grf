############################################################################
# RUN MCMC WITH VECCHIA MODEL
############################################################################

library(parallel)

#----------------------------------------------------------------

## Load function to create Vecchia model
source("setupVecchiaTemp.R")

#----------------------------------------------------------------

# Define MCMC parameters
nChains=1
ntot_iter=75000
nRuns=1
thinning=100
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
id_call=args[5] # Id of the call

#--------------------------------------------------------------
## File paths

## Id file and export folder
if(is.na(subsamp)){
  id_file=paste0("run_",id_sim,id_mask,"_v",kVecchia)
}else{
  id_file=paste0("run_",id_sim,id_mask,"_v",kVecchia,"_s",subsamp)
}
export_folder=paste0(path_files,id_file,"/")
if(!dir.exists(export_folder)){
  dir.create(export_folder)
}
id_file=paste0(id_file,"_")

#--------------------------------------------------------------

cat("MCMC data: Data Id=",id_sim,"/ Mask Id=",id_mask,"/ Vecchia Approx=",kVecchia,"/ Subsamp=",subsamp,"\n")

#--------------------------------------------------------------


## Ordering
orderingFile=paste0(path_files,id_file,"obs_ordering.rds")
if(file.exists(orderingFile)){
  sgvOrdering=readRDS(orderingFile)
}else{
  stop("Make ordering first")
}


cat("Creating Vecchia Model...\n")

res=createVecchiaModel(path_files,id_file,id_sim,id_mask,kVecchia,tau2obs,subsamp,
                       sgvOrdering=sgvOrdering)
Rmodel=res$model
Yobs=res$data
obsCoord=res$obsCoord
nodeMat=res$allCoord
sel_mask=res$indObs
knot_blocks=res$knot_blocks
rm(res)

#--------------------------------------------------------------

cat("Setting up Vecchia MCMC...\n")

conf <- configureMCMC(Rmodel,control = list(scale=0.05,adaptative=TRUE))
conf$removeSamplers()
## Global range
conf$addSampler(target = "Sigma_coef1", type = "RW", silent = TRUE)
## Anisotropy ratio
conf$addSampler(target = "Sigma_coef2", type = "RW", silent = TRUE)
## Spline knots
for(i in 1:length(knot_blocks)){
  conf$addSampler(target = paste0("Sigma_coef3[",knot_blocks[[i]],"]"), type = "RW_block", silent = TRUE)
}

Rmcmc <- buildMCMC(conf)
Cmodel <- compileNimble(Rmodel)
Cmcmc <- compileNimble(Rmcmc, project = Rmodel)



#--------------------------------------------------------------

cat("Running Vecchia MCMC...\n")


fit <- list(runMCMC(Cmcmc, niter = ntot_iter, nburnin = 0,progressBar=FALSE,
               thin=thinning))
# Save checkpoint
saveRDS(fit, paste0(export_folder,id_file, id_call, "_" ,i, ".rds"))

# Update initial values for next batch
final_values <- as.matrix(do.call(rbind,lapply(fit,function(x){matrix(x,ncol=36+2)[nrow(x),]})))
write.csv(final_values,paste0(export_folder,id_file, id_call, "_" ,i, ".csv"),row.names = F)

cat("End of Vecchia MCMC.\n")





