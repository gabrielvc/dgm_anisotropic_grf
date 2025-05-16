#' Pullback metric coefficients
#'
#' Computes the coefficients of a metric matrix defined by pullback at a given triangle.
#'
#'@param p1,p2,p3 Coordinates of the points defining the triangle.
#'@param Gfunc Function returning the 3x3 metric matrix to be pulled back at a given point.
#'
#'@return The coefficients of the matrix of the pulled-back metric on the triangle.
#'
#'@keywords internal
#'
.triMetricPullCoefs<-function(p1,p2,p3,Gfunc){
  Jf=cbind(p2-p1,p3-p1)
  G=t(Jf)%*%Gfunc((p1+p2+p3)/3)%*%Jf
  return(c(G[1,1],G[1,2],G[2,2]))
}
.triMetricPullCoefs1d<-function(p1,p2,Gfunc){
  Jf=cbind(p2-p1)
  G=t(Jf)%*%Gfunc((p1+p2)/2)%*%Jf
  return(c(G))
}


#' Pullback metric coefficients for 2D surface
#'
#' Computes the coefficients of a metric matrix defined by pullback for each triangle of a triangulation.
#'
#'@param nodeMat Coordinates of the points defining the triangle.
#'@param triMat Indices of the nodes forming each triangle.
#'@param Gfunc Function returning the metric matrix to be pulled back at a given point. If NULL, the Euclidean metric is used.
#'@param useParallel Whether to parallelize the computations using the `parallel` package. Default = TRUE.
#'@param mc.cores Number of cores to use when using parallel computations. Default = NULL means that the number of detected cores is used (or the default value in the parallel package if the detection fails).
#'
#'@return The coefficients of the matrix of the pulled-back metric for each triangle.
#'
#'
triMetricPull2d<-function(nodeMat,triMat,Gfunc=NULL,useParallel=TRUE, mc.cores=NULL,verbose=FALSE){
  if(verbose) message("Computing metric coefficients for each triangle...")
  if(is.null(Gfunc)){
    Gf<-function(p){
      return(diag(ncol(nodeMat)))
    }
  }else{
    Gf=Gfunc
    testG=Gf(nodeMat[1,])
    if(sum(abs(dim(testG)-rep(ncol(nodeMat),2)))!=0){
      stop(paste0("The metric function should return a ",ncol(nodeMat),"x",ncol(nodeMat)," matrix."))
    }
  }
  if(useParallel){
    res=t(mcsapply(1:nrow(triMat),function(i){.triMetricPullCoefs(nodeMat[triMat[i,1],],nodeMat[triMat[i,2],],nodeMat[triMat[i,3],],Gf)},mc.cores))
  }else{
    res=t(apply(triMat,1,function(v){.triMetricPullCoefs(nodeMat[v[1],],nodeMat[v[2],],nodeMat[v[3],],Gf)}))
  }
  if(verbose) message("Done.")
  return(res)
}


#' Pullback metric coefficients for 1D surface
#'
#' Computes the coefficients of a metric matrix defined by pullback for each segment of the discretization of a 1D surface.
#'
#'@param nodeMat Coordinates of the nodes defining the segments.
#'@param triMat Indices of the nodes forming each segment.
#'@param Gfunc Function returning the metric matrix to be pulled back at a given point. If NULL, the Euclidean metric is used.
#'@param useParallel Whether to parallelize the computations using the `parallel` package. Default = TRUE.
#'@param mc.cores Number of cores to use when using parallel computations. Default = NULL means that the number of detected cores is used (or the default value in the parallel package if the detection fails).
#'
#'@return The coefficients of the matrix of the pulled-back metric for each triangle.
#'
#'
triMetricPull1d<-function(nodeMat,triMat,Gfunc=NULL,useParallel=TRUE, mc.cores=NULL,verbose=FALSE){
  if(verbose) message("Computing metric coefficients for each triangle...")
  if(is.null(Gfunc)){
    Gf<-function(p){
      return(diag(ncol(nodeMat)))
    }
  }else{
    Gf=Gfunc
    testG=Gf(nodeMat[1,])
    if(sum(abs(dim(testG)-rep(ncol(nodeMat),2)))!=0){
      stop(paste0("The metric function should return a ",ncol(nodeMat),"x",ncol(nodeMat)," matrix."))
    }
  }
  if(useParallel){
    res=matrix(t(mcsapply(1:nrow(triMat),function(i){.triMetricPullCoefs1d(nodeMat[triMat[i,1],],nodeMat[triMat[i,2],],Gf)},mc.cores)),ncol=1)
  }else{
    res=matrix(c(apply(triMat,1,function(v){.triMetricPullCoefs1d(nodeMat[v[1],],nodeMat[v[2],],Gf)})),ncol=1)
  }
  if(verbose) message("Done.")
  return(res)
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#' Spherical coordinates
#'
#' Computes spherical coordinates from Cartesian coordinates.
#'
#'@param x0 Cartesian coordinates of the point.
#'
#'@return Vector containing the spherical coordinates.
#'
coordSph<-function(x0){
  x=x0/(x0[1]**2+x0[2]**2+x0[3]**2)**0.5
  rho=(x[1]**2+x[2]**2)**0.5
  theta=acos(x[3])
  if(rho==0){
    return(c(theta,pi/2))
  }
  phi=atan(x[2]/x[1])+(x[1]<0)*(2*(x[2]>=0)-1)*pi
  return(c(theta,phi))
}



#' Spherical metric coefficients
#'
#' Computes the coefficients of a metric matrix defined from spherical coordinates at a given triangle. Only for a UNIT SPHERE.
#'
#'@param p1,p2,p3 Coordinates of the points defining the triangle.
#'@param GfuncSph Function returning the 2x2 metric matrix in spherical coordinates.
#'
#'@return The coefficients of the metric matrix on the triangle.
#'
#'@keywords internal
#'
.triMetricSphCoefs<-function(p1,p2,p3,GfuncSph){

  Jf=cbind(p2-p1,p3-p1)

  x=(p1+p2+p3)/3
  r=(x[1]**2+x[2]**2+x[3]**2)**0.5
  rho=(x[1]**2+x[2]**2)**0.5
  Jpsi=cbind(c(x[1]*x[3]/(rho*r**2),-x[2]/rho**2),
             c(x[2]*x[3]/(rho*r**2),x[1]/rho**2),
             c(-rho/r**2,0))

  xSph=coordSph(x)

  G=t(Jf)%*%t(Jpsi)%*%GfuncSph(xSph)%*%Jpsi%*%Jf
  return(c(G[1,1],G[1,2],G[2,2]))
}


#' Spherical metric coefficients
#'
#' Computes the coefficients of a metric matrix defined from spherical coordinates for each triangle of a triangulation. Only for a UNIT SPHERE.
#'
#'@param nodeMat Coordinates of the points defining the triangle.
#'@param triMat Indices of the nodes forming each triangle.
#'@param GfuncSph Function returning the 2x2 metric matrix in spherical coordinates. If NULL, the canonical metric on the sphere is used.
#'@param useParallel Whether to parallelize the computations using the `parallel` package. Default = TRUE.
#'@param mc.cores Number of cores to use when using parallel computations. Default = NULL means that the number of detected cores is used (or the default value in the parallel package if the detection fails).
#'
#'@return The coefficients of the metric matrix for each triangle.
#'
#'
triMetricSph<-function(nodeMat,triMat,GfuncSph=NULL,useParallel=TRUE, mc.cores=NULL){
  message("OBS: This method should only be used for a UNIT SPHERE.")
  message("Computing metric coefficients for each triangle...")
  if(is.null(GfuncSph)){
    Gf<-function(p){
      return(matrix(c(1,0,0,sin(p[1])**2),nrow=2))
    }
  }else{
    Gf=GfuncSph
    testG=Gf(coordSph(nodeMat[1,]))
    if(sum(abs(dim(testG)-rep(2,2)))!=0){
      stop("The metric function should return a 2x2 matrix.")
    }
  }
  if(useParallel){
    res=t(mcsapply(1:nrow(triMat),function(i){.triMetricSphCoefs(nodeMat[triMat[i,1],],nodeMat[triMat[i,2],],nodeMat[triMat[i,3],],Gf)},mc.cores))
  }else{
    res=t(apply(triMat,1,function(v){.triMetricSphCoefs(nodeMat[v[1],],nodeMat[v[2],],nodeMat[v[3],],Gf)}))
  }
  message("Done.")
  return(res)
}


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#' Spherical diffusion or metric tensor coefficients
#'
#' Compute coefficients of the diffusion or metric tensor on a triangle, for a sphere
#'
#'@param p1,p2,p3 Coordinates of the points defining the triangle.
#'@param dir Angle giving the principal direction of diffusion in the tangent space of the sphere.
#'@param rho1 vector giving the relative weight of the principal diffusion direction.
#'@param rho2 vector giving the relative weight of the secondary diffusion direction.
#'
#'@return The coefficients of the diffusion or metric tensor for the  triangle.
#'
.computeTriDiff<-function(p1,p2,p3,dir,rho1,rho2){

  ## Compute change of variable Jacobians
  Jf=cbind(p2-p1,p3-p1)
  x=(p1+p2+p3)/3
  r=(x[1]**2+x[2]**2+x[3]**2)**0.5
  rho=(x[1]**2+x[2]**2)**0.5
  Jpsi=cbind(c(x[1]*x[3]/(rho*r**2),-x[2]/rho**2),
             c(x[2]*x[3]/(rho*r**2),x[1]/rho**2),
             c(-rho/r**2,0))

  ## Compute spherical coordinates
  xSph=coordSph(x/sqrt(sum(x**2)))
  x=xSph

  ## Direction
  v=c(cos(dir),sin(dir))

  ## Compute the diffusion directions
  u=c(v[1],v[2])*sin(x[1])/((sin(x[1])**2*v[1]**2+1*v[2]**2))**0.5
  uorth=c(-v[2],(sin(x[1])**2)*v[1])*1.0/((sin(x[1])**2)*v[1]**2+v[2]**2)**0.5

  rhoH=c(rho1,rho2)
  H=rhoH[1]*rhoH[2]*((1/rhoH[1])*u%*%t(u)+(1/rhoH[2])*uorth%*%t(uorth))
  G=t(Jf)%*%t(Jpsi)%*%H%*%Jpsi%*%Jf
  return(c(G[1,1],G[1,2],G[2,2]))
}


#' Spherical diffusion or metric tensor coefficients
#'
#' Compute coefficients of the diffusion or metric tensor on each triangle of a mesh, for a sphere
#'
#'@param nodeMat Coordinates of the points defining the triangle.
#'@param triMat Indices of the nodes forming each triangle.
#'@param dirNode Angle giving the principal direction of diffusion in the tangent space of each node of the sphere.
#'@param rhoNode Matrix giving the relative weights of the diffusion directions at each node.
#'
#'@return The coefficients of the diffusion or metric tensor for each triangle.
#'
triMetricFromValues<-function(nodeMat,triMat,dirNode,rhoNode){

  ## Compute directions and weights on each triangle
  xdirTri=rowMeans(cos(cbind(dirNode[triMat[,1]],dirNode[triMat[,2]],dirNode[triMat[,3]])))
  ydirTri=rowMeans(sin(cbind(dirNode[triMat[,1]],dirNode[triMat[,2]],dirNode[triMat[,3]])))
  dirTri=atan2(ydirTri,xdirTri)
  # dirTri=(dirNode[triMat[,1]]+dirNode[triMat[,2]]+dirNode[triMat[,3]])/3

  rhoTri=(rhoNode[triMat[,1],]+rhoNode[triMat[,2],]+rhoNode[triMat[,3],])/3

  ## Compute coefficients of the diffusion tensor on each triangle
  matTri=cbind(triMat,dirTri,rhoTri)

  if(useParallel){
    triDiffMat=t(mcsapply(1:nrow(matTri), function(i){.computeTriDiff(nodeMat[matTri[i,1],],nodeMat[matTri[i,2],],nodeMat[matTri[i,3],],matTri[i,4],matTri[i,5],matTri[i,6])},mc.cores))
  }else{
    triDiffMat=t(apply(matTri, 1, function(v){.computeTriDiff(nodeMat[v[1],],nodeMat[v[2],],nodeMat[v[3],],v[4],v[5],v[6])}))
  }


  return(triDiffMat)
}


