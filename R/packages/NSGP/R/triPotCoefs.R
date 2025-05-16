
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#' Potential function coefficients
#'
#' Computes the coefficients of a potential function.
#'
#'@param nodeMat Coordinates of the points defining the triangle.
#'@param triMat Indices of the nodes forming each triangle.
#'@param potf Potential function. If set to a (positive) constant, the corresponding constant function is used. Otherwise, it should be a positive function of the coordinates.
#'@param coordSph Logical, whether the potential function is defined as function of the spherical coordinates (Default=FALSE).
#'@param useParallel Whether to parallelize the computations using the `parallel` package. Default = TRUE.
#'@param mc.cores Number of cores to use when using parallel computations. Default = NULL means that the number of detected cores is used (or the default value in the parallel package if the detection fails).
#'
#'@return The coefficients of the potential function for each triangle.
#'
triPot<-function(nodeMat,triMat,potf=1,coordSph=FALSE,useParallel=TRUE, mc.cores=NULL){
  message("Computing potential function coefficients for each triangle...")
  if((class(potf)=="numeric")&&(potf>0)){
    res=(rep(potf,nrow(triMat)))
  }else{
    ## Coordinates of the barycenter of each triangle
    barTriCoord=(nodeMat[triMat[,1],]+nodeMat[triMat[,2],]+nodeMat[triMat[,3],])/3 ## Cartesian
    if(coordSph){
      barTriCoord=t(apply(barTriCoord, 1, coordSph))  ## Spherical
    }
    ## Potential function on each triangle
    if(useParallel){
      res=mcsapply(1:nrow(barTriCoord),function(i){potf(barTriCoord[i,])},mc.cores)
    }else{
      res=apply(barTriCoord,1,potf)
    }
  }
  message("Done.")
  return(res)
}




