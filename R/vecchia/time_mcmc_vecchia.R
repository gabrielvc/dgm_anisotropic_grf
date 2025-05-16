############################################################################
# RUN MCMC WITH VECCHIA MODEL
############################################################################

library(NSGP)
library(Matrix)
library(rhdf5)
library(BayesNSGP)

#----------------------------------------------------------------

## Load function to create Vecchia model

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




#----------------------------------------------------------------

# Define MCMC parameters
nChains=1
ntot_iter=10
nRuns=1
thinning=1
tau2obs=0.05**2

nmm=5

nruns=1
nmmp=5

path_files="./vecchia_temp/"

timed=TRUE

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

npost=ifelse(kVecchia==64,1,10)
ntot_iter=ifelse(kVecchia==64,3,30)


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

#--------------------------------------------------------------
cat("Ordering...\n")

tictoc::tic()
for(i in 1:nmm){
  sgvOrdering <- sgvSetup(coords = obsCoord, k = kVecchia, 
                          seed=sample(1e5,1))
}
tm_ord=tictoc::toc()

t1=(tm_ord$toc-tm_ord$tic)/nmm
t2=NA
t3=NA
t4=NA
uu=matrix(c(t1,t2,
            t3,t4),nrow=1)
colnames(uu)=c("Ordering","MCMC","Ordering_post","Post")
write.csv(uu,paste0(path_files,"time",id_file, id_call, ".csv"))

#-----------------------------------------------------



cat("Creating Vecchia Model...\n")

tictoc::tic()

Rmodel <- nsgpModelSimple(likelihood = "SGV", 
                          mu_model = "zero",
                          Sigma_model = "custom",
                          tau_model="fixed",
                          sigma_model = "fixed",
                          constants = constants, 
                          coords = constants$coords,
                          data = Yobs,
                          constInit=F,
                          sgvOrdering=sgvOrdering)

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

tm_compmod=tictoc::toc()

t1b=(tm_compmod$toc-tm_compmod$tic)

uu=matrix(c(t1,t1b,t2,
            t3,t4),nrow=1)
colnames(uu)=c("Ordering","Model Compliation","MCMC","Ordering_post","Post")
write.csv(uu,paste0(path_files,"time",id_file, id_call, ".csv"))



#--------------------------------------------------------------

cat("Running Vecchia MCMC...\n")

tictoc::tic()
# Run MCMC in batches
run <- runMCMC(Cmcmc, niter = ntot_iter, nburnin = 0,progressBar=TRUE,
                 thin=thinning)
tm_mcmc=tictoc::toc()

cat("End of Vecchia MCMC.\n")


t2=(tm_mcmc$toc-tm_mcmc$tic)/(ntot_iter)
uu=matrix(c(t1,t1b,t2,
            t3,t4),nrow=1)
colnames(uu)=c("Ordering","Model Compliation","MCMC","Ordering_post","Post")
write.csv(uu,paste0(path_files,"time",id_file, id_call, ".csv"))

#--------------------------------------------------------------

vnames=c("Sigma_coef1","Sigma_coef2","Sigma_coef3[1]","Sigma_coef3[2]","Sigma_coef3[3]","Sigma_coef3[4]","Sigma_coef3[5]","Sigma_coef3[6]","Sigma_coef3[7]","Sigma_coef3[8]","Sigma_coef3[9]","Sigma_coef3[10]","Sigma_coef3[11]","Sigma_coef3[12]","Sigma_coef3[13]","Sigma_coef3[14]","Sigma_coef3[15]","Sigma_coef3[16]","Sigma_coef3[17]","Sigma_coef3[18]","Sigma_coef3[19]","Sigma_coef3[20]","Sigma_coef3[21]","Sigma_coef3[22]","Sigma_coef3[23]","Sigma_coef3[24]","Sigma_coef3[25]","Sigma_coef3[26]","Sigma_coef3[27]","Sigma_coef3[28]","Sigma_coef3[29]","Sigma_coef3[30]","Sigma_coef3[31]","Sigma_coef3[32]","Sigma_coef3[33]","Sigma_coef3[34]","Sigma_coef3[35]","Sigma_coef3[36]")
param_list=matrix(c(0.140362852930756,0.966368878616716,-19.022038137699,3.76382908381281,-12.7237437535235,5.39960385064841,7.15385838102745,-2.31581360308316,8.10706646839086,-8.34795253580197,-14.8117841724737,-2.17560626911884,-8.04140537911218,1.59367501256055,-4.31815262187129,-11.4834401794887,1.52787533596162,-15.7235831324277,-2.569985004636,2.19831253509873,6.2823875422152,10.6523220931784,-2.65264548299834,18.644205148543,2.93620237841913,-12.6156567254147,-8.32776772370812,-5.37401513665154,11.602895372195,7.74082200083325,6.95087775779356,6.54378678102526,-2.79239339898919,-5.8800710372986,2.97684766593168,1.69096793746828,6.23811452762385,-9.04977193406508
),nrow=1)
param_list=t(matrix(rep(param_list,npost),ncol=npost))
npost=nrow(param_list)

#--------------------------------------------------------------

## Prediction locations
sel_pred=setdiff(1:nrow(nodeMat),sel_mask)
Xpred=nodeMat[sel_pred,]
matPredSplines=setSplineCovarMatrix(Xpred,xknots)
constants=list()
constants[["PX_Sigma"]]=matPredSplines
ordObs=Rmodel$getConstants()$ord

tictoc::tic()
for(i in 1:nmmp){
  sgvOrdering <- sgvSetup(coords = obsCoord[ordObs,], coords_pred = Xpred, k = kVecchia, 
                          seed=sample(1e5,1),order_coords = FALSE)
}
tm_ordp=tictoc::toc()

t3=(tm_ordp$toc-tm_ordp$tic)/nmmp
uu=matrix(c(t1,t1b,t2,
            t3,t4),nrow=1)
colnames(uu)=c("Ordering","Model Compliation","MCMC","Ordering_post","Post")
write.csv(uu,paste0(path_files,"time",id_file, id_call, ".csv"))



cnt=1
tictoc::tic()
## Select samples
spred=param_list
colnames(spred)=vnames
## predict at target locations
uu=nsgpPredict(Rmodel,samples = spred,coords.predict = Xpred,
                 sgvOrdering=sgvOrdering,
                 constants = constants)
tm_post=tictoc::toc()

t4=(tm_post$toc-tm_post$tic)/(npost)

uu=matrix(c(t1,t1b,t2,
            t3,t4),nrow=1)
colnames(uu)=c("Ordering","Model Compliation","MCMC","Ordering_post","Post")
write.csv(uu,paste0(path_files,"time",id_file, id_call, ".csv"))
