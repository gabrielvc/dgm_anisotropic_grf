#' Parallel sapply function
#'
#' Parallel version of the sapply function, using the mclapply function from the \code{parallel} package.
#'
#'@param X Vector to which the sapply function if to be applied.
#'@param FUN Function to apply to each entry of the vector.
#'@param mc.cores Number of cores to use. Default = NULL means that the number of detected cores is used (or the default value in the parallel package if the detection fails).
#'
#'@return Result of the sapply function.
#'
#'
mcsapply <- function (X, FUN, mc.cores=NULL,...) {
  mc.cores=ifelse(is.null(mc.cores),parallel::detectCores(),mc.cores)
  mc.cores = ifelse(is.na(mc.cores),getOption("mc.cores",2),mc.cores)
  res = parallel::mclapply(X = X, FUN = FUN,mc.cores=mc.cores,...)
  if(length(res)>0){
    res=simplify2array(res)
  }
  return(res)
}


#' Plot on mesh of 2D surface
#'
#' 3D plot a variable defined on each node of a mesh using Rgl.
#'
#'@param zplt Vector containing the values, at each mesh node, of the variable to be plotted
#'@param msh Mesh object: Object of class "mesh3d" obtained after importing a mesh through the `vcgImport` function of the `Rvcg` package.
#'@param col Vector of colors defining the palette used for the plot (Default=NULL corresponds to a Spectral palette with 192 colors).
#'@param ttl Title of the plot.
#'@param zlim : Range of values used for color bar: "range" = use the range of values of zplt (Default), "sym" = use a symetric range, or put a vector of size 2 with a custom range.
#'@param windowSize : Size of the (square) plotting window.
#'
#'@return Rgl plot of the mesh with colors given by the variable.
#'
#'
plotOnMesh<-function(zplt,msh,col=NULL,ttl="",zlim="range",windowSize=800){

  if(is.null(col)){
    ## Spectral palette with 128 colors
    col=c('#B500A3','#B202A4','#B005A5','#AE07A6','#AC0AA7','#AA0DA8','#A80FAA',
          '#A612AB','#A415AC','#A217AD','#A01AAE','#9E1CAF','#9C1FB1','#9A22B2',
          '#9824B3','#9627B4','#942AB5','#922CB6','#902FB8','#8E32B9','#8C34BA',
          '#8A37BB','#8739BC','#853CBE','#833FBF','#8141C0','#7F44C1','#7D47C2',
          '#7B49C3','#794CC5','#774FC6','#7551C7','#7354C8','#7156C9','#6F59CA',
          '#6D5CCC','#6B5ECD','#6961CE','#6764CF','#6566D0','#6369D1','#616CD3',
          '#5F6ED4','#5C71D5','#5A73D6','#5876D7','#5679D9','#547BDA','#527EDB',
          '#5081DC','#4E83DD','#4C86DE','#4A89E0','#488BE1','#468EE2','#4490E3',
          '#4293E4','#4096E5','#3E98E7','#3C9BE8','#3A9EE9','#38A0EA','#36A3EB',
          '#34A6ED','#34A6ED','#3BA9E8','#43ACE3','#4AB0DF','#52B3DA','#59B7D5',
          '#61BAD1','#68BECC','#70C1C7','#77C5C3','#7FC8BE','#87CCBA','#8ECFB5',
          '#96D3B0','#9DD6AC','#A5DAA7','#ACDDA4','#B1DFA6','#B6E1A7','#BCE4A9',
          '#C1E6AB','#C6E8AD','#CCEAAE','#D1ECB0','#D6EEB2','#DCF0B3','#E1F3B5',
          '#E6F5B7','#ECF7B9','#F1F9BA','#F6FBBC','#FCFDBE','#FEFCBC','#FEF7B6',
          '#FEF2B0','#FEEDAA','#FEE7A4','#FEE29E','#FEDD98','#FED892','#FDD38C',
          '#FDCE86','#FDC980','#FDC37A','#FDBE74','#FDB96E','#FDB468','#FDAF62',
          '#FCA85E','#FCA05C','#FB9859','#FB9056','#FB8853','#FA8050','#FA784E',
          '#FA704B','#F96848','#F96045','#F85842','#F85040','#F8483D','#F7403A',
          '#F73837','#F73135','#F73135','#F43034','#F12F33','#EF2E32','#EC2E31',
          '#E92D30','#E72C30','#E42B2F','#E22B2E','#DF2A2D','#DC292C','#DA282C',
          '#D7282B','#D4272A','#D22629','#CF2528','#CD2528','#CA2427','#C72326',
          '#C52225','#C22224','#C02124','#BD2023','#BA1F22','#B81F21','#B51E20',
          '#B21D1F','#B01C1F','#AD1C1E','#AB1B1D','#A81A1C','#A5191B','#A3191B',
          '#A0181A','#9D1719','#9B1618','#981617','#961517','#931416','#901315',
          '#8E1314','#8B1213','#891113','#861012','#831011','#810F10','#7E0E0F',
          '#7B0D0E','#790D0E','#760C0D','#740B0C','#710A0B','#6E0A0A','#6C090A',
          '#690809','#660708','#640707','#610606','#5F0506','#5C0405','#590404',
          '#570303','#540202','#520202')
  }

  ncolPal=length(col)
  if(zlim=="range"){
    zlim=range(zplt)
  }else if(zlim=="sym"){
    zlim=max(abs(zplt))*c(-1,1)
  }else if(length(zlim)==2){
    zlim=zlim
  }else{
    zlim=range(zplt)
  }

  brks=unique(c(seq(from=zlim[1],to=zlim[2],length.out=ncolPal+1)))
  col.vec=col[cut(zplt,breaks = ncolPal,labels = F)]
  shade3d(msh,meshColor = "vertices",color=col.vec,
          edges="black",
          material=list(lit=F))
  par3d(windowRect = c(0, 0, windowSize, windowSize))
  bgplot3d({image.plot(legend.only = TRUE, zlim = range(zplt), col = col,legend.width = 0.8)
           title(main = ttl, line = 0,cex.main=3)})

}



#' Plot on mesh of 1D surface
#'
#' 2D plot a variable defined on each node of the mesh  of a 1D surface.
#'
#'@param nodeVals Vector containing the values, at each mesh node, of the variable to be plotted
#'@param nodeMat Coordinates of the nodes defining the segments.
#'@param triMat Indices of the nodes forming each segment.
#'@param col Vector of colors defining the palette used for the plot (Default=NULL corresponds to a Spectral palette with 192 colors).
#'
#'@param ttl Title of the plot.
#'@param zlim : Range of values used for color bar: "range" = use the range of values of zplt (Default), "sym" = use a symetric range, or put a vector of size 2 with a custom range.
#'
#'@return Plot of the mesh with colors given by the variable.
#'
#'
plot1d<-function(nodeVals,nodeMat,triMat,colorPlot=FALSE,r=0.3,col=NULL,ttl="",zlim="range",...){

  if(is.null(col)){
    ## Spectral palette with 128 colors
    col=c('#B500A3','#B202A4','#B005A5','#AE07A6','#AC0AA7','#AA0DA8','#A80FAA',
          '#A612AB','#A415AC','#A217AD','#A01AAE','#9E1CAF','#9C1FB1','#9A22B2',
          '#9824B3','#9627B4','#942AB5','#922CB6','#902FB8','#8E32B9','#8C34BA',
          '#8A37BB','#8739BC','#853CBE','#833FBF','#8141C0','#7F44C1','#7D47C2',
          '#7B49C3','#794CC5','#774FC6','#7551C7','#7354C8','#7156C9','#6F59CA',
          '#6D5CCC','#6B5ECD','#6961CE','#6764CF','#6566D0','#6369D1','#616CD3',
          '#5F6ED4','#5C71D5','#5A73D6','#5876D7','#5679D9','#547BDA','#527EDB',
          '#5081DC','#4E83DD','#4C86DE','#4A89E0','#488BE1','#468EE2','#4490E3',
          '#4293E4','#4096E5','#3E98E7','#3C9BE8','#3A9EE9','#38A0EA','#36A3EB',
          '#34A6ED','#34A6ED','#3BA9E8','#43ACE3','#4AB0DF','#52B3DA','#59B7D5',
          '#61BAD1','#68BECC','#70C1C7','#77C5C3','#7FC8BE','#87CCBA','#8ECFB5',
          '#96D3B0','#9DD6AC','#A5DAA7','#ACDDA4','#B1DFA6','#B6E1A7','#BCE4A9',
          '#C1E6AB','#C6E8AD','#CCEAAE','#D1ECB0','#D6EEB2','#DCF0B3','#E1F3B5',
          '#E6F5B7','#ECF7B9','#F1F9BA','#F6FBBC','#FCFDBE','#FEFCBC','#FEF7B6',
          '#FEF2B0','#FEEDAA','#FEE7A4','#FEE29E','#FEDD98','#FED892','#FDD38C',
          '#FDCE86','#FDC980','#FDC37A','#FDBE74','#FDB96E','#FDB468','#FDAF62',
          '#FCA85E','#FCA05C','#FB9859','#FB9056','#FB8853','#FA8050','#FA784E',
          '#FA704B','#F96848','#F96045','#F85842','#F85040','#F8483D','#F7403A',
          '#F73837','#F73135','#F73135','#F43034','#F12F33','#EF2E32','#EC2E31',
          '#E92D30','#E72C30','#E42B2F','#E22B2E','#DF2A2D','#DC292C','#DA282C',
          '#D7282B','#D4272A','#D22629','#CF2528','#CD2528','#CA2427','#C72326',
          '#C52225','#C22224','#C02124','#BD2023','#BA1F22','#B81F21','#B51E20',
          '#B21D1F','#B01C1F','#AD1C1E','#AB1B1D','#A81A1C','#A5191B','#A3191B',
          '#A0181A','#9D1719','#9B1618','#981617','#961517','#931416','#901315',
          '#8E1314','#8B1213','#891113','#861012','#831011','#810F10','#7E0E0F',
          '#7B0D0E','#790D0E','#760C0D','#740B0C','#710A0B','#6E0A0A','#6C090A',
          '#690809','#660708','#640707','#610606','#5F0506','#5C0405','#590404',
          '#570303','#540202','#520202')
  }

  plot(rbind(apply(nodeMat,2,min),apply(nodeMat,2,max)),cex=0,xlab="x",ylab="y",asp=1,main=ttl)

  zplt=(nodeVals[triMat[,1]]+nodeVals[triMat[,1]])/2

  ncolPal=length(col)
  if(zlim=="range"){
    zlim=range(zplt)
  }else if(zlim=="sym"){
    zlim=max(abs(zplt))*c(-1,1)
  }else if(length(zlim)==2){
    zlim=zlim
  }else{
    zlim=range(zplt)
  }

  brks=unique(c(seq(from=zlim[1],to=zlim[2],length.out=ncolPal+1)))
  col.vec=col[cut(zplt,breaks = ncolPal,labels = F)]
  res=apply(as.matrix(1:nrow(triMat)),1,function(i){lines(nodeMat[triMat[i,],1],nodeMat[triMat[i,],2],col=col.vec[i],...); return(0)})

  return(NULL)
}




#' Plot variable as deformed 1D surface
#'
#' 2D plot of a variable defined on each node of the mesh of a 1D surface, as a deformation of the surface.
#' DISCLAIMER: Only valid for meshes where the neighbors of a node i are i-1 and i+1.
#'
#'@param nodeVals Vector containing the values, at each mesh node, of the variable to be plotted
#'@param nodeMat Coordinates of the nodes defining the segments.
#'@param triMat Indices of the nodes forming each segment.
#'@param r Scaling size for the radial deformation (only used if `colorPlot=FALSE`). The deformed surface, plotted in green, is obtained by translating each node along the normal direction by a length r.
#'@param ttl Title of the plot.
#'@param ... Additional arguments to R base `plot` function
#'
#'@return Plot of the mesh (grey) and of the radially deformed mesh (green).
#'
#'
plot1d_deform<-function(nodeVals,nodeMat,triMat,r=0.1,ttl="",...){

  plot((1+r)*rbind(apply(nodeMat,2,min),apply(nodeMat,2,max)),cex=0,asp=1,xlab="x",ylab="y",main=ttl,...)
  vmax=max(abs(nodeVals))

  normvec<-function(i,i1,i2){
    nn=c(-(nodeMat[i2,2]-nodeMat[i1,2]),(nodeMat[i2,1]-nodeMat[i1,1]))
    return(nn/sum(nn**2)**0.5)
  }

  imax=nrow(nodeMat)
  nf<-function(i){
    (r*nodeVals[i]/vmax)*normvec(i,max(1,i-1),min(i+1,imax))
  }

  normVecMat=t(sapply(1:imax, nf))
  coordplt=nodeMat + normVecMat

  res=apply(as.matrix(1:nrow(triMat)),1,function(i){lines(nodeMat[triMat[i,],1],nodeMat[triMat[i,],2],col="grey50"); return(0)})
  res=apply(as.matrix(1:nrow(triMat)),1,function(i){lines(coordplt[triMat[i,],1],coordplt[triMat[i,],2],lwd=1.5, col = "forestgreen"); return(0)})
  return(NULL)
}



