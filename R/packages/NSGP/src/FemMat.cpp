// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Sparse>

#include "Tools.h"


typedef Eigen::Triplet<double> T;


//' Mass and stiffness matrices for 2D surfaces
//'
//' Computes the mass and stiffness matrices associated with a second-order elliptic symmetric operator \eqn{L=-\text{div}(A \nabla) + V}.
//'
//'@param nodeMat Array containing the coordinates of the triangulation nodes
//'@param triMat Array containing, for each triangle, the node indices (starting at 1).
//'@param triMetricMat Array containing, for each triangle, the coefficients of the metric tensor of each triangle.
//'@param triPotCoeffs Array containing, for each triangle, the value of the potential function \eqn{V}. If set to \code{NULL} (Default), no potential function is considered.
//'@param triDiffCoeffs Array containing, for each triangle, the coefficients of the modified metric tensor associated with the diffusion matrix \eqn{A}. If set to \code{NULL} (Default), the usual metric is considered (which corresponds to the case where \eq{A} is the identity matrix).
//'@param massLumping logical; TRUE (Default) = Compute mass lumping approximation of mass matrix, FALSE = Compute true mass matrix.
//'
//'@details We consider second-order elliptic symmetric operators \eqn{L} of the form \deqn{L=-\text{div}(A \nabla) + V} where \eqn{V} denotes a so-called potential function taking postive values,
//' \eqn{A} denotes a (linear) operator mapping tangent vectors at a given point to tangent vectors at the same point, \eqn{\test{div}} denotes the divergence operator, and \eqn{\nabla} denotes the gradient operator.
//'
//'@details The function returns the mass and stiffness matrices associated with linear finite elements on a mesh. The mass matrix \eqn{C} contains entries \eqn{[(\psi_i, \psi_j)]_{1 \le i,j \le n}},
//' the mass lumped matrix  \eqn{\hat{C}} is diagonal with entries \eqn{[(\psi_i, 1)]_{1 \le i\le n}},
//' the stiffness matrix \eqn{R} contains entries \eqn{[(V\psi_i, \psi_j)+(A\nabla\psi_i, \nabla\psi_j)]_{1 \le i,j \le n}}
//' The scaled stiffness matrix is defined as \eqn{S=\hat{C}^{-1/2}R\hat{C}^{-1/2}}.
//'
//'@details The coefficients in \code{triMetricMat} can be computed using the functions \code{triMetricSph} and \code{triMetricPull}.
//' The coefficients in \code{triPotCoeffs} can be computed using the function \code{triPot}.
//'
//'@details The coefficients in \code{triDiffCoeffs} correspond to the overall tensor, in local coordinates, that allows us to compute the
//'\eqn{(A\nabla\psi_i, \nabla\psi_j)} using the gradients of \eqn{\psi_i} and \eqn{\psi_j} as computed in the reference simplex. Such coefficients can be computed using the functions \code{triMetricSph} and \code{triMetricPull}.
//'
//'@return List with entries:
//' \describe{
//'   \item{\code{Shift}}{Scaled stiffness matrix \eqn{S}}
//'   \item{\code{Scale}}{Scaling matrix \eqn{\hat{C}^{-1/2}}}
//'   \item{\code{Eig}}{Interval containing the eigenvalues of \eqn{S}}
//'   \item{\code{Mass}}{Mass matrix \eqn{C} or its mass-lumped approximation  \eqn{\hat{C}} (stored as an array if mass lumping is applied, and a sparse matrix otherwise)}
//'   \item{\code{Stiff}}{Stiffness matrix \eqn{R}}
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List matFEM2d(Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat,
                  Eigen::ArrayXXd& triMetricMat,
                  Rcpp::Nullable<Rcpp::NumericVector > triPotCoeffs=R_NilValue,
                  Rcpp::Nullable<Rcpp::NumericMatrix > triDiffCoeffs=R_NilValue,
                  bool massLumping=true){

  // Check NULL argument
  bool varKappa=false;
  Eigen::ArrayXd triCoeffs;
  if (!triPotCoeffs.isNull()) {
    varKappa=true;
    triCoeffs=Rcpp::as<Eigen::Map<Eigen::ArrayXd>>(triPotCoeffs.as());
  }
  bool varDiff=false;
  Eigen::ArrayXXd triDiffOpMat;
  if (!triDiffCoeffs.isNull()) {
    varDiff=true;
    triDiffOpMat=Rcpp::as<Eigen::Map<Eigen::ArrayXXd>>(triDiffCoeffs.as());
  }

  // Check argument sizes
  if((nodeMat.cols()<2)||(triMat.cols()!=3)){
    throw std::invalid_argument("The node coordinates matrix should have 2 or more columns, and the matrix containing the triangle nodes should have 3 colmuns.");
  }
  if(triMat.maxCoeff()!=nodeMat.rows()){
    throw std::invalid_argument("Inconsistency between the indices in the triangle node matrix and the total number of nodes.");
  }
  if(triMat.rows()!=triMetricMat.rows()){
    throw std::invalid_argument("The number of anistropy matrix coefficents must be the same as the number of triangles.");
  }
  if(varKappa){
    if(triMat.rows()!=triCoeffs.size()){
      throw std::invalid_argument("The number of potential function coefficents must be the same as the number of triangles.");
    }
  }

  double g11,g12,g22,detG,hT,hTm1,ml,tck,bck,normGrad;
  int n0,n1,n2;
  int nbNodes=triMat.maxCoeff();

  Eigen::ArrayXd Cdiag=Eigen::ArrayXd::Zero(nbNodes);

  Eigen::ArrayXd nbTriForNode=Eigen::ArrayXd::Zero(nbNodes);

  Eigen::SparseMatrix<double> R(nbNodes,nbNodes);
  Eigen::ArrayXd Jvec;

  // Triplets for sparse matrix construction
  std::vector<T> tripletListStiff,tripletListStiffKappa, tripletListMass;
  tripletListStiff.reserve(9*triMat.rows());
  if(varKappa){
    tripletListStiffKappa.reserve(9*triMat.rows());
  }
  if(!massLumping){
    tripletListMass.reserve(9*triMat.rows());
  }

  for(int k=0; k<triMat.rows();++k){

    // Indices of the triangle nodes
    n0=triMat(k,0)-1;
    n1=triMat(k,1)-1;
    n2=triMat(k,2)-1;

    // Metric coefficients
    g11=triMetricMat(k,0);
    g12=triMetricMat(k,1);
    g22=triMetricMat(k,2);


    // Deteminant of the metric
    detG=g11*g22-g12*g12;
    hT=std::pow(detG,0.5);
    hTm1=std::pow(detG,-0.5);



    // Elements of the lumped mass matrix: (psi_i, 1)= hT/(d+1)!
    ml=hT/6.0;
    Cdiag(n0)+=ml;
    Cdiag(n1)+=ml;
    Cdiag(n2)+=ml;

    // Elements of the Potential function matrix
    if(varKappa){
      tck=triCoeffs(k)*hT/24.0;
      tripletListStiffKappa.push_back(T(n0,n0,2*tck));
      tripletListStiffKappa.push_back(T(n1,n0,tck));
      tripletListStiffKappa.push_back(T(n0,n1,tck));
      tripletListStiffKappa.push_back(T(n1,n1,2*tck));
      tripletListStiffKappa.push_back(T(n2,n0,tck));
      tripletListStiffKappa.push_back(T(n0,n2,tck));
      tripletListStiffKappa.push_back(T(n2,n1,tck));
      tripletListStiffKappa.push_back(T(n1,n2,tck));
      tripletListStiffKappa.push_back(T(n2,n2,2*tck));
    }


    // Elements of the true mass matrix
    bck=hT/24.0;
    nbTriForNode(n0)+=1;
    nbTriForNode(n1)+=1;
    nbTriForNode(n2)+=1;
    if(!massLumping){
      tripletListMass.push_back(T(n0,n0,2*bck));
      tripletListMass.push_back(T(n1,n0,bck));
      tripletListMass.push_back(T(n0,n1,bck));
      tripletListMass.push_back(T(n1,n1,2*bck));
      tripletListMass.push_back(T(n2,n0,bck));
      tripletListMass.push_back(T(n0,n2,bck));
      tripletListMass.push_back(T(n2,n1,bck));
      tripletListMass.push_back(T(n1,n2,bck));
      tripletListMass.push_back(T(n2,n2,2*bck));
    }

    // Compute Jmat
    if(varDiff){
      Jvec=triDiffOpMat.row(k);
      g11=Jvec(0);
      g12=Jvec(1);
      g22=Jvec(2);
    }


    // Elements of the stiffness matrix: (nabla psi_i, nabla psi_j)=(e_{ki}^t G^{-1} e_{kj})hT/d!
    tripletListStiff.push_back(T(n0,n0,hTm1*(g11-2*g12+g22)/2.0));
    tripletListStiff.push_back(T(n1,n0,hTm1*(-g22+g12)/2.0));
    tripletListStiff.push_back(T(n0,n1,hTm1*(-g22+g12)/2.0));
    tripletListStiff.push_back(T(n1,n1,hTm1*g22/2.0));
    tripletListStiff.push_back(T(n0,n2,hTm1*(g12-g11)/2.0));
    tripletListStiff.push_back(T(n2,n0,hTm1*(g12-g11)/2.0));
    tripletListStiff.push_back(T(n1,n2,hTm1*(-g12)/2.0));
    tripletListStiff.push_back(T(n2,n1,hTm1*(-g12)/2.0));
    tripletListStiff.push_back(T(n2,n2,hTm1*g11/2.0));


  }

  // Eigenvalues interval
  Eigen::ArrayXd approxInterval(2);
  approxInterval(0)=0;

  // Build stiffness matrix
  R.setFromTriplets(tripletListStiff.begin(), tripletListStiff.end());
  R.makeCompressed();

  // Add contribution of varying kappa
  Eigen::SparseMatrix<double> CvarKappa(nbNodes,nbNodes);
  if(varKappa){
    CvarKappa.setFromTriplets(tripletListStiffKappa.begin(), tripletListStiffKappa.end());
    CvarKappa.makeCompressed();
    R=R+CvarKappa;
    approxInterval(0)=(triCoeffs.minCoeff())*(6.0/24.0)*nbTriForNode.minCoeff(); // bck/ml
  }

  // Build diagonal sparse matrix C^(-1/2) using mass lumping
  std::vector<T> tripletListCmhML;
  tripletListCmhML.reserve(nbNodes);
  for(int i=0; i<nbNodes; ++i){
    tripletListCmhML.push_back(T(i,i,pow(Cdiag(i),-0.5)));
  }
  Eigen::SparseMatrix<double> CmhML(nbNodes,nbNodes);
  CmhML.setFromTriplets(tripletListCmhML.begin(), tripletListCmhML.end());
  CmhML.makeCompressed();
  
  std::vector<T> tripletListCmhMLInv;
  tripletListCmhMLInv.reserve(nbNodes);
  for(int i=0; i<nbNodes; ++i){
    tripletListCmhMLInv.push_back(T(i,i,pow(Cdiag(i),0.5)));
  }
  Eigen::SparseMatrix<double> CmhMLInv(nbNodes,nbNodes);
  CmhMLInv.setFromTriplets(tripletListCmhMLInv.begin(), tripletListCmhMLInv.end());
  CmhMLInv.makeCompressed();


  // Scaled stiffness matrix
  Eigen::SparseMatrix<double> S=CmhML*R*CmhML;

  // Compute max eigenvalues
  Eigen::ArrayXd vecTemp=Eigen::ArrayXd::Zero(nbNodes);
  for (int k=0; k<S.outerSize(); ++k){
    for (Eigen::SparseMatrix<double>::InnerIterator it(S,k); it; ++it){
      vecTemp(k)+=std::abs(it.value());
    }
  }
  approxInterval(1)=vecTemp.maxCoeff();


  if(!massLumping){
    // Build non lumped mass matrix
    Eigen::SparseMatrix<double> C(nbNodes,nbNodes);
    C.setFromTriplets(tripletListMass.begin(), tripletListMass.end());
    C.makeCompressed();

    return Rcpp::List::create(Rcpp::_["Shift"]=S, Rcpp::_["Scale"]=CmhML, Rcpp::_["Eig"]=approxInterval,
                              Rcpp::_["Mass"]=C, Rcpp::_["Stiff"]=R,Rcpp::_["ScaleInv"]=CmhMLInv);

  }else{

    return Rcpp::List::create(Rcpp::_["Shift"]=S, Rcpp::_["Scale"]=CmhML, Rcpp::_["Eig"]=approxInterval,
                              Rcpp::_["Mass"]=CvarKappa, Rcpp::_["Stiff"]=R,Rcpp::_["ScaleInv"]=CmhMLInv);

  }
}




//' Mass and stiffness matrices for 1D surfaces
//'
//' Computes the mass and stiffness matrices associated with a second-order elliptic symmetric operator \eqn{L=-\text{div}(A \nabla) + V}.
//'
//'@param nodeMat Array containing the coordinates of the triangulation nodes
//'@param triMat Array containing, for each triangle, the node indices (starting at 1).
//'@param triMetricMat Array containing, for each triangle, the coefficients of the metric tensor of each triangle.
//'@param triPotCoeffs Array containing, for each triangle, the value of the potential function \eqn{V}. If set to \code{NULL} (Default), no potential function is considered.
//'@param triDiffCoeffs Array containing, for each triangle, the coefficients of the modified metric tensor associated with the diffusion matrix \eqn{A}. If set to \code{NULL} (Default), the usual metric is considered (which corresponds to the case where \eq{A} is the identity matrix).
//'@param massLumping logical; TRUE (Default) = Compute mass lumping approximation of mass matrix, FALSE = Compute true mass matrix.
//'
//'@details We consider second-order elliptic symmetric operators \eqn{L} of the form \deqn{L=-\text{div}(A \nabla) + V} where \eqn{V} denotes a so-called potential function taking postive values,
//' \eqn{A} denotes a (linear) operator mapping tangent vectors at a given point to tangent vectors at the same point, \eqn{\test{div}} denotes the divergence operator, and \eqn{\nabla} denotes the gradient operator.
//'
//'@details The function returns the mass and stiffness matrices associated with linear finite elements on a mesh. The mass matrix \eqn{C} contains entries \eqn{[(\psi_i, \psi_j)]_{1 \le i,j \le n}},
//' the mass lumped matrix  \eqn{\hat{C}} is diagonal with entries \eqn{[(\psi_i, 1)]_{1 \le i\le n}},
//' the stiffness matrix \eqn{R} contains entries \eqn{[(V\psi_i, \psi_j)+(A\nabla\psi_i, \nabla\psi_j)]_{1 \le i,j \le n}}
//' The scaled stiffness matrix is defined as \eqn{S=\hat{C}^{-1/2}R\hat{C}^{-1/2}}.
//'
//'@details The coefficients in \code{triMetricMat} can be computed using the function \code{triMetricPull1d}.
//'
//'@details Since we are in 1D, \code{triDiffCoeffs} and \code{triPotCoeffs} are just matrices with a single column and positive values.
//'
//'@details The coefficients in \code{triDiffCoeffs} correspond to the overall tensor, in local coordinates, that allows us to compute the
//'\eqn{(A\nabla\psi_i, \nabla\psi_j)} using the gradients of \eqn{\psi_i} and \eqn{\psi_j} as computed in the reference simplex.
//'
//'@return List with entries:
//' \describe{
//'   \item{\code{Shift}}{Scaled stiffness matrix \eqn{S}}
//'   \item{\code{Scale}}{Scaling matrix \eqn{\hat{C}^{-1/2}}}
//'   \item{\code{Eig}}{Interval containing the eigenvalues of \eqn{S}}
//'   \item{\code{Mass}}{Mass matrix \eqn{C} or its mass-lumped approximation  \eqn{\hat{C}} (stored as an array if mass lumping is applied, and a sparse matrix otherwise)}
//'   \item{\code{Stiff}}{Stiffness matrix \eqn{R}}
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List matFEM1d(Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat,
                  Eigen::ArrayXXd& triMetricMat,
                  Rcpp::Nullable<Rcpp::NumericVector > triPotCoeffs=R_NilValue,
                  Rcpp::Nullable<Rcpp::NumericMatrix > triDiffCoeffs=R_NilValue,
                  bool massLumping=true){

  // Check NULL argument
  bool varKappa=false;
  Eigen::ArrayXd triCoeffs;
  if (!triPotCoeffs.isNull()) {
    varKappa=true;
    triCoeffs=Rcpp::as<Eigen::Map<Eigen::ArrayXd>>(triPotCoeffs.as());
  }
  bool varDiff=false;
  Eigen::ArrayXXd triDiffOpMat;
  if (!triDiffCoeffs.isNull()) {
    varDiff=true;
    triDiffOpMat=Rcpp::as<Eigen::Map<Eigen::ArrayXXd>>(triDiffCoeffs.as());
  }

  // Check argument sizes
  if((nodeMat.cols()<2)||(triMat.cols()!=2)){
    throw std::invalid_argument("The node coordinates matrix should have 1 or more columns, and the matrix containing the triangle nodes should have 2 colmuns.");
  }
  if(triMat.maxCoeff()!=nodeMat.rows()){
    throw std::invalid_argument("Inconsistency between the indices in the triangle node matrix and the total number of nodes.");
  }
  if(triMat.rows()!=triMetricMat.rows()){
    throw std::invalid_argument("The number of anistropy matrix coefficents must be the same as the number of triangles.");
  }
  if(varKappa){
    if(triMat.rows()!=triCoeffs.size()){
      throw std::invalid_argument("The number of potential function coefficents must be the same as the number of triangles.");
    }
  }

  double g11,detG,hT,hTm1,ml,tck,bck,normGrad;
  int n0,n1;
  int nbNodes=triMat.maxCoeff();
  double adiff;


  Eigen::ArrayXd Cdiag=Eigen::ArrayXd::Zero(nbNodes);
  Eigen::ArrayXd nbTriForNode=Eigen::ArrayXd::Zero(nbNodes);


  Eigen::SparseMatrix<double> R(nbNodes,nbNodes);
  Eigen::ArrayXd Jvec;

  // Triplets for sparse matrix construction
  std::vector<T> tripletListStiff,tripletListStiffKappa, tripletListMass;
  tripletListStiff.reserve(9*triMat.rows());
  if(varKappa){
    tripletListStiffKappa.reserve(9*triMat.rows());
  }
  if(!massLumping){
    tripletListMass.reserve(9*triMat.rows());
  }

  for(int k=0; k<triMat.rows();++k){

    // Indices of the triangle nodes
    n0=triMat(k,0)-1;
    n1=triMat(k,1)-1;

    // Metric coefficients
    g11=triMetricMat(k,0);


    // Deteminant of the metric
    detG=std::abs(g11);
    hT=std::pow(detG,0.5);
    hTm1=std::pow(detG,-0.5);



    // Elements of the lumped mass matrix: (psi_i, 1)= hT/(d+1)!
    ml=hT/2.0;
    Cdiag(n0)+=ml;
    Cdiag(n1)+=ml;

    // Elements of the Potential function matrix
    if(varKappa){
      tck=triCoeffs(k)*hT/6.0;
      tripletListStiffKappa.push_back(T(n0,n0,2*tck));
      tripletListStiffKappa.push_back(T(n1,n0,tck));
      tripletListStiffKappa.push_back(T(n0,n1,tck));
      tripletListStiffKappa.push_back(T(n1,n1,2*tck));
    }


    // Elements of the true mass matrix
    nbTriForNode(n0)+=1;
    nbTriForNode(n1)+=1;
    if(!massLumping){
      bck=hT/6.0;
      tripletListMass.push_back(T(n0,n0,2*bck));
      tripletListMass.push_back(T(n1,n0,bck));
      tripletListMass.push_back(T(n0,n1,bck));
      tripletListMass.push_back(T(n1,n1,2*bck));
    }

    // Compute Jmat
    if(varDiff){
      adiff=triDiffOpMat(k,0);
    }else{
      adiff=1.0;
    }

    // Elements of the stiffness matrix: (nabla psi_i, nabla psi_j)=(e_{ki}^t G^{-1} e_{kj})hT/d!
    tripletListStiff.push_back(T(n0,n0,hTm1*adiff/1.0));
    tripletListStiff.push_back(T(n1,n0,hTm1*(-adiff)/1.0));
    tripletListStiff.push_back(T(n0,n1,hTm1*(-adiff)/1.0));
    tripletListStiff.push_back(T(n1,n1,hTm1*adiff/1.0));

  }

  // Eigenvalues interval
  Eigen::ArrayXd approxInterval(2);
  approxInterval(0)=0;

  // Build stiffness matrix
  R.setFromTriplets(tripletListStiff.begin(), tripletListStiff.end());
  R.makeCompressed();

  // Add contribution of varying kappa
  Eigen::SparseMatrix<double> CvarKappa(nbNodes,nbNodes);
  if(varKappa){
    CvarKappa.setFromTriplets(tripletListStiffKappa.begin(), tripletListStiffKappa.end());
    CvarKappa.makeCompressed();
    R=R+CvarKappa;
    approxInterval(0)=(triCoeffs.minCoeff())*(2.0/6.0)*nbTriForNode.minCoeff(); // bck/ml
  }


  // Build diagonal sparse matrix C^(-1/2) using mass lumping
  std::vector<T> tripletListCmhML;
  tripletListCmhML.reserve(nbNodes);
  for(int i=0; i<nbNodes; ++i){
    tripletListCmhML.push_back(T(i,i,pow(Cdiag(i),-0.5)));
  }
  Eigen::SparseMatrix<double> CmhML(nbNodes,nbNodes);
  CmhML.setFromTriplets(tripletListCmhML.begin(), tripletListCmhML.end());
  CmhML.makeCompressed();


  // Scaled stiffness matrix
  Eigen::SparseMatrix<double> S=CmhML*R*CmhML;

  // Compute max eigenvalues
  Eigen::ArrayXd vecTemp=Eigen::ArrayXd::Zero(nbNodes);
  for (int k=0; k<S.outerSize(); ++k){
    for (Eigen::SparseMatrix<double>::InnerIterator it(S,k); it; ++it){
      vecTemp(k)+=std::abs(it.value());
    }
  }
  approxInterval(1)=vecTemp.maxCoeff();


  if(!massLumping){
    // Build non lumped mass matrix
    Eigen::SparseMatrix<double> C(nbNodes,nbNodes);
    C.setFromTriplets(tripletListMass.begin(), tripletListMass.end());
    C.makeCompressed();

    return Rcpp::List::create(Rcpp::_["Shift"]=S, Rcpp::_["Scale"]=CmhML, Rcpp::_["Eig"]=approxInterval,
                              Rcpp::_["Mass"]=C, Rcpp::_["Stiff"]=R);

  }else{

    return Rcpp::List::create(Rcpp::_["Shift"]=S, Rcpp::_["Scale"]=CmhML, Rcpp::_["Eig"]=approxInterval,
                              Rcpp::_["Mass"]=CvarKappa, Rcpp::_["Stiff"]=R);

  }
}
