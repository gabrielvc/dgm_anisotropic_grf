############################################################################
# simulation test on a square
############################################################################

library(NSGP)
library(RTriangle)
library(Matrix)
library(fields)
library(rhdf5)

args <- commandArgs(trailingOnly = TRUE)
id_file=as.numeric(args[1])
if(sum(is.na(c(id_file)))>0){
  stop("Two numerical argumuments are expected")
}
suffix_name=args[2]
foldername=paste0(suffix_name,"/")
if(!is.na(suffix_name)){
  filename=paste0(foldername,"data_",suffix_name,"_",id_file,".h5")
}else{
  filename=paste0("data_",id_file,".h5")
}

set.seed(3000+id_file*100)

gen_sim<-function(theta,xknots,nodeMat,triMat,nsim=1){
  
  param_f=theta[1:nknots]
  param_rho_scale=theta[nknots+1:2]
  a=theta[nknots+3]
  nu=theta[nknots+4]
  # tau2=theta[3*nknots+3]
  

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
  
  return(z_sim)
  
}


#--------------------------------------------------------------

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

## Delaunay triangulation
res=RTriangle::triangulate(pslg(nodeMat),S=0)
nodeMat=res$P
triMat=res$T
triMetricMat=triMetricPull2d(nodeMat,triMat)
rm(res)

## Final number of nodes
N=nrow(nodeMat)
Nf=Ngd*Ngd

#----------------------------------

## Knots
nknots_grid=6
xspl=seq(from=min(nodeMat),to=max(nodeMat),length.out=nknots_grid)
xknots=as.matrix(expand.grid(xspl,xspl))
nknots=nrow(xknots)

nparams=nknots+4

#---------------------------------------

## Number of examples
nex=5000
nsim=5

## Create file
export_folder="./"
filename=paste0(export_folder,filename )
if (file.exists(filename)) {
  #Delete file if it exists
  file.remove(filename)
}
h5createFile(filename)
h5createDataset(filename, "data", rev(c(nsim,1,Ngd,Ngd)), maxdims = rev(c(nex*nsim,1,Ngd,Ngd)),
                chunk=rev(c(1,1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "maps", rev(c(nsim,1,Ngd,Ngd)), maxdims = rev(c(nex*nsim,1,Ngd,Ngd)),
                chunk=rev(c(1,2,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "params", rev(c(nsim,nparams)),maxdims = rev(c(nex*nsim,nparams)),
                native=TRUE)
cnt=1
for(k in 1:nex){
  print(paste0("Configuration ",k,"/",nex))
  
  # Anisotropy directions
  param_f=rnorm(nknots,0,1)
  
  # Varaible scale
  amin=0.05; amax=0.3
  a=runif(1,amin,amax)

  ## Anistropy ratio
  param_rho_scale=c(runif(1,0.1,1),1)
  if(runif(1)>0.5){
    param_rho_scale=rev(param_rho_scale)
  }
  
  # Regualrity : FIXED
  nu=2
  
  ## Store parameters
  params=c(param_f,param_rho_scale,a,nu)

  ############################################################################
  
  ## Simulation
  z_sim=gen_sim(params,xknots,nodeMat,triMat,nsim=nsim)
  z_sim=z_sim[ind_sel,]
  
  ## Parameters map
  f_map=evalTPSpline(nodeMat,xknots,param_f,m=3,p=2)[ind_sel]
  
  # Store
  for (i in 1:ncol(z_sim)) {
    h5write(matrix(z_sim[,i],Ngd,Ngd), 
            file=filename, name="data", index=list(1:Ngd,1:Ngd,1,cnt))
    h5write(matrix(f_map,Ngd,Ngd), 
            file=filename, name="maps", index=list(1:Ngd,1:Ngd,1,cnt))
    h5write(params, file=filename, name="params", index=list(1:nparams,cnt))
    cnt=cnt+1
  }
  
  if(k<nex){
    h5set_extent(filename, "data",c(Ngd,Ngd,1,cnt-1+nsim))
    h5set_extent(filename, "maps",c(Ngd,Ngd,1,cnt-1+nsim))
    h5set_extent(filename, "params",c(nparams,cnt-1+nsim))
  }
  
}
h5closeAll()
