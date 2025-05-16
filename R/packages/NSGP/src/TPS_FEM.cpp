// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>

#include "ThinPlateSpline.h"
#include "ChebCoeffsFFT.h"
#include "FemMat.h"
#include "simKrigTools.h"
#include "Tools.h"


Eigen::Matrix2d Gfunc(Eigen::ArrayXd& pt, 
                      TPS& T_f, TPS& T_rho1, TPS& T_rho2,double rho_min,double rho_max){
  
  // Gradient
  Eigen::Vector2d u = T_f.eval_grad(pt(0),pt(1));
  u=u/u.norm();
  Eigen::Vector2d u_orth;
  u_orth(0)=-u(1);
  u_orth(1)=u(0);
  
  // Scalings
  Eigen::Vector2d rho;
  // rho(0)=rho_min+(1-rho_min)/(1+std::exp(-T_rho1.eval(pt(0),pt(1))));
  // rho(1)=rho_min+(1-rho_min)/(1+std::exp(-T_rho2.eval(pt(0),pt(1))));
  rho(0)=lgt(T_rho1.eval(pt(0),pt(1)),rho_min,rho_max);
  rho(1)=lgt(T_rho2.eval(pt(0),pt(1)),rho_min,rho_max);
  
  Eigen::Matrix2d res;
  res=(1.0/(rho(0)*rho(0)))*u*u.transpose() + (1.0/(rho(1)*rho(1)))*u_orth*u_orth.transpose() ;
  return res;
}

Eigen::Matrix2d GfuncConstRho(Eigen::ArrayXd& pt, 
                      TPS& T_f, double rho1, double rho2){
  
  // Gradient
  Eigen::Vector2d u = T_f.eval_grad(pt(0),pt(1));
  u=u/u.norm();
  Eigen::Vector2d u_orth;
  u_orth(0)=-u(1);
  u_orth(1)=u(0);
  
  // Scalings
  Eigen::Vector2d rho;
  // rho(0)=rho_min+(1-rho_min)/(1+std::exp(-T_rho1.eval(pt(0),pt(1))));
  // rho(1)=rho_min+(1-rho_min)/(1+std::exp(-T_rho2.eval(pt(0),pt(1))));
  rho(0)=rho1;
  rho(1)=rho2;
  
  Eigen::Matrix2d res;
  res=(1.0/(rho(0)*rho(0)))*u*u.transpose() + (1.0/(rho(1)*rho(1)))*u_orth*u_orth.transpose() ;
  return res;
}



Eigen::ArrayXXd triMetricCompute2d(Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat, 
                                   TPS& T_f, TPS& T_rho1, TPS& T_rho2, double rho_min=0.001,
                                   double rho_max=1){
  
  Eigen::ArrayXXd res = Eigen::ArrayXXd::Zero(triMat.rows(),3);
  if((nodeMat.cols()!= 2)||(triMat.cols()!=3)){
    Rcpp::Rcout<<"Wrond dimensions for nodeMat or triMat: the function triMetricCompute2d only works in the 2D Euclidean case.";
    return res;
  }
  
  
  Eigen::ArrayXi triInd(3);
  Eigen::Matrix2d Gmat, tJGmatJ, J;
  Eigen::ArrayXd p1(2), p2(2), p3(2), bar(2);
    
  for(int i=0; i< res.rows(); ++i){
    triInd = triMat.row(i)-1;
    p1=nodeMat.row(triInd(0));
    p2=nodeMat.row(triInd(1));
    p3=nodeMat.row(triInd(2));
    bar=(p1+p2+p3)/3;
    
    Gmat = Gfunc(bar, T_f, T_rho1, T_rho2,rho_min,rho_max);
    J.col(0)=p2-p1;
    J.col(1)=p3-p1;
    tJGmatJ=(J.transpose()) * Gmat * J;
    
    res(i,0)=tJGmatJ(0,0);
    res(i,1)=tJGmatJ(1,0);
    res(i,2)=tJGmatJ(1,1);
  }
  
  return res;
  
}


Eigen::ArrayXXd triMetricCompute2dConstRho(Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat, 
                                   TPS& T_f, double rho1, double rho2){
  
  Eigen::ArrayXXd res = Eigen::ArrayXXd::Zero(triMat.rows(),3);
  if((nodeMat.cols()!= 2)||(triMat.cols()!=3)){
    Rcpp::Rcout<<"Wrond dimensions for nodeMat or triMat: the function triMetricCompute2d only works in the 2D Euclidean case.";
    return res;
  }
  
  
  Eigen::ArrayXi triInd(3);
  Eigen::Matrix2d Gmat, tJGmatJ, J;
  Eigen::ArrayXd p1(2), p2(2), p3(2), bar(2);
  
  for(int i=0; i< res.rows(); ++i){
    triInd = triMat.row(i)-1;
    p1=nodeMat.row(triInd(0));
    p2=nodeMat.row(triInd(1));
    p3=nodeMat.row(triInd(2));
    bar=(p1+p2+p3)/3;
    
    Gmat = GfuncConstRho(bar, T_f, rho1, rho2);
    J.col(0)=p2-p1;
    J.col(1)=p3-p1;
    tJGmatJ=(J.transpose()) * Gmat * J;
    
    res(i,0)=tJGmatJ(0,0);
    res(i,1)=tJGmatJ(1,0);
    res(i,2)=tJGmatJ(1,1);
  }
  
  return res;
  
}



// [[Rcpp::export]]
Eigen::ArrayXXd triMetricCompute2d(Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat, 
                                   Eigen::ArrayXXd& knots, 
                                   Eigen::ArrayXd& val_f, 
                                   Eigen::ArrayXd val_rho1= Eigen::ArrayXd::Ones(1), 
                                   Eigen::ArrayXd val_rho2= Eigen::ArrayXd::Ones(1), 
                                   int m=3, int p=2, double lambda=0, double rho_min=0.001,
                                   double rho_max=1){
  
  // Gradient
  TPS T_f(knots,val_f,m,p,lambda);
  // Scalings
  
  if(val_rho1.size()>1){
    TPS T_rho1(knots,val_rho1,m,p,lambda);
    TPS T_rho2(knots,val_rho2,m,p,lambda);
    return triMetricCompute2d(nodeMat, triMat, 
                              T_f, T_rho1,  T_rho2, rho_min,rho_max);
  }else{
    return triMetricCompute2dConstRho(nodeMat, triMat, 
                              T_f, val_rho1(0),val_rho2(0));
  }
}


Eigen::Matrix2d Gfunc2d(Eigen::ArrayXd& pt, 
                                   Eigen::ArrayXXd& knots, 
                                   Eigen::ArrayXd& val_f, Eigen::ArrayXd& val_rho1, Eigen::ArrayXd& val_rho2, 
                                   int m, int p=2, double lambda=0, double rho_min=0.001,double rho_max=1){
  
  // Gradient
  TPS T_f(knots,val_f,m,p,lambda);
  // Scalings
  TPS T_rho1(knots,val_rho1,m,p,lambda);
  TPS T_rho2(knots,val_rho2,m,p,lambda);
  
 return Gfunc(pt, T_f,  T_rho1, T_rho2, rho_min,rho_max);
  
}

//--------------------------------------------------------------------


// Log likelihood
// [[Rcpp::export]]
double loglikelihood(Eigen::ArrayXd& theta, Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A,
                     Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat,
                     Eigen::ArrayXXd& knots,
                     int m=3, int p=2, double lambda=0, double rho_min=0.001){

  int nknots = knots.rows();

  Eigen::ArrayXd param_f = theta.segment(0, nknots);
  Eigen::ArrayXd param_rho1 = theta.segment(nknots, nknots+nknots);
  Eigen::ArrayXd param_rho2 = theta.segment(2*nknots,2*nknots+ nknots);
  double a = theta(3*nknots);
  double tau2 = theta(3*nknots + 2);

  // double nu = theta(3*nknots+1);
  double nu = 1;

  Eigen::ArrayXXd triMetricMat = triMetricCompute2d(nodeMat,triMat,
                                     knots,
                                     param_f, param_rho1, param_rho2,
                                     m, p, lambda, rho_min);

  Rcpp::List FEMatList = matFEM2d(nodeMat,  triMat, triMetricMat);


  double phi = a/std::sqrt(8*nu);
  double phi2 = phi*phi;
  double c2 = 1/(4*std::pow(std::tgamma(0.5)*phi,2)*nu);

  int n = nodeMat.rows();
  Eigen::SparseMatrix<double> I(n, n);
  I.setIdentity();

  Eigen::SparseMatrix<double> S = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["Shift"]);
  Eigen::SparseMatrix<double> Ch = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["ScaleInv"]);
  Eigen::SparseMatrix<double> Q = c2*Ch*(I + phi2*S)*(I + phi2*S)*Ch;
  Eigen::SparseMatrix<double> Qsum = Q + (1/tau2)*(A.transpose())*A;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > cholQ(Q);
  Eigen::ArrayXd vectD=cholQ.vectorD();
  double logdetQ = vectD.log().sum();

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > cholQsum(Qsum);
  vectD=cholQsum.vectorD();
  double logdetQsum = vectD.log().sum();

  Eigen::VectorXd Ytilde = (A.transpose())*Y;
  double tYY = Y.array().square().sum();


  Eigen::VectorXd solveY = cholQsum.solve(Ytilde);
  double YsolveY = ((Ytilde.array())*(solveY.array())).sum();

  int nobs = Y.size();
  double l = -0.5*(nobs*std::log(tau2)+logdetQsum-logdetQ+tYY/tau2-YsolveY/(tau2*tau2));

  return l;


}



// //-----------------------------------------------------------
// 
// Eigen::MatrixXd NCsampleBase(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQ,
//                              Eigen::MatrixXd& W){
//   
//   Eigen::ArrayXd vectDmh=cholQ.vectorD().array().sqrt().inverse();
//   Eigen::MatrixXd res(W.rows(),W.cols());
//   for(int i=0; i<W.cols(); ++i){
//     res.col(i)= cholQ.permutationPinv()*(cholQ.matrixU()).solve((vectDmh* W.col(i).array()).matrix());
//   }
//   
//   return res;
//   
// }
// 
// 
// Eigen::VectorXd KrigBase(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQsum,
//                             Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A, double tau2){
//   
//   
//   Eigen::VectorXd Ytilde = (A.transpose())*Y;
//   Eigen::VectorXd solveY = cholQsum.solve(Ytilde);
//   
//   return (1.0/tau2)*solveY;
//   
// }
// 
// Eigen::MatrixXd CsampleBase(Eigen::VectorXd& Zk,
//                           Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQ,
//                         Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQsum,
//                         Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A, double tau2,
//                         Eigen::MatrixXd& W, Eigen::MatrixXd& eps){
// 
//   // Eigen::VectorXd Zk=KrigBase(cholQsum,Y,A, tau2);
//   
//   Eigen::MatrixXd Znc = NCsampleBase(cholQ, W);
//   Eigen::MatrixXd Znck(Znc.rows(),Znc.cols());
//   for(int i=0; i<W.cols(); ++i){
//     Eigen::VectorXd Ync = A * Znc.col(i) + std::sqrt(tau2)*eps.col(i);
//     Znck.col(i)=Zk + Znc.col(i) - KrigBase(cholQsum, Ync, A, tau2);
//   }
//   
//   return Znck;
//   
// }

//-----------------------------------------------------------

//Kriging
// [[Rcpp::export]]
Eigen::VectorXd KrigNodes(Eigen::ArrayXd& theta, Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A,
                     Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat,
                     Eigen::ArrayXXd& knots,
                     int m=3, int p=2, double lambda=0, double rho_min=0.001){
  
  int nknots = knots.rows();
  
  Eigen::ArrayXd param_f = theta.segment(0, nknots);
  Eigen::ArrayXd param_rho1 = theta.segment(nknots, nknots+nknots);
  Eigen::ArrayXd param_rho2 = theta.segment(2*nknots,2*nknots+ nknots);
  double a = theta(3*nknots);
  double tau2 = theta(theta.size()-1);
  // double nu = theta(3*nknots+1);
  double nu = 1;
  
  Eigen::ArrayXXd triMetricMat = triMetricCompute2d(nodeMat,triMat,
                                                    knots,
                                                    param_f, param_rho1, param_rho2,
                                                    m, p, lambda, rho_min);
  
  Rcpp::List FEMatList = matFEM2d(nodeMat,  triMat, triMetricMat);
  
  double phi = a/std::sqrt(8*nu);
  double phi2 = phi*phi;
  double c2 = 1/(4*std::pow(std::tgamma(0.5)*phi,2)*nu);
  
  int n = nodeMat.rows();
  Eigen::SparseMatrix<double> I(n, n);
  I.setIdentity();
  
  Eigen::SparseMatrix<double> S = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["Shift"]);
  Eigen::SparseMatrix<double> Ch = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["ScaleInv"]);
  Eigen::SparseMatrix<double> Q = c2*Ch*(I + phi2*S)*(I + phi2*S)*Ch;
  Eigen::SparseMatrix<double> Qsum = Q + (1/tau2)*(A.transpose())*A;
  
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > cholQ(Q);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > cholQsum(Qsum);
  
  
  Eigen::VectorXd Zk=KrigBase(cholQsum,Y,A, tau2);
  
  return Zk;
  
}



// Non conditional simulation
// [[Rcpp::export]]
Eigen::MatrixXd NCondSampleNodes(Eigen::ArrayXd& theta, Eigen::MatrixXd& W,
                           Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat,
                           Eigen::ArrayXXd& knots,
                           int m=3, int p=2, double lambda=0, double rho_min=0.001){
  
  int nknots = knots.rows();
  
  Eigen::ArrayXd param_f = theta.segment(0, nknots);
  Eigen::ArrayXd param_rho1 = theta.segment(nknots, nknots+nknots);
  Eigen::ArrayXd param_rho2 = theta.segment(2*nknots,2*nknots+ nknots);
  double a = theta(3*nknots);
  double tau2 = theta(theta.size()-1);
  // double nu = theta(3*nknots+1);
  double nu = 1;
  
  Eigen::ArrayXXd triMetricMat = triMetricCompute2d(nodeMat,triMat,
                                                    knots,
                                                    param_f, param_rho1, param_rho2,
                                                    m, p, lambda, rho_min);
  
  Rcpp::List FEMatList = matFEM2d(nodeMat,  triMat, triMetricMat);
  
  double phi = a/std::sqrt(8*nu);
  double phi2 = phi*phi;
  double c2 = 1/(4*std::pow(std::tgamma(0.5)*phi,2)*nu);
  
  int n = nodeMat.rows();
  Eigen::SparseMatrix<double> I(n, n);
  I.setIdentity();
  
  Eigen::SparseMatrix<double> S = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["Shift"]);
  Eigen::SparseMatrix<double> Ch = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["ScaleInv"]);
  Eigen::SparseMatrix<double> Q = c2*Ch*(I + phi2*S)*(I + phi2*S)*Ch;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > cholQ(Q);

  
  return NCsampleBase( cholQ,W);
  
}


// Conditional simulation
// [[Rcpp::export]]
Eigen::MatrixXd CondSampleNodes(Eigen::ArrayXd& theta, Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A,
                             Eigen::MatrixXd& W, Eigen::MatrixXd& eps,
                            Eigen::ArrayXXd& nodeMat, Eigen::ArrayXXi& triMat,
                     Eigen::ArrayXXd& knots,
                     int m=3, int p=2, double lambda=0, double rho_min=0.001){
  
  int nknots = knots.rows();
  
  Eigen::ArrayXd param_f = theta.segment(0, nknots);
  Eigen::ArrayXd param_rho1 = theta.segment(nknots, nknots+nknots);
  Eigen::ArrayXd param_rho2 = theta.segment(2*nknots,2*nknots+ nknots);
  double a = theta(3*nknots);
  double tau2 = theta(theta.size()-1);
  // double nu = theta(3*nknots+1);
  double nu = 1;
  
  Eigen::ArrayXXd triMetricMat = triMetricCompute2d(nodeMat,triMat,
                                                    knots,
                                                    param_f, param_rho1, param_rho2,
                                                    m, p, lambda, rho_min);
  
  Rcpp::List FEMatList = matFEM2d(nodeMat,  triMat, triMetricMat);
  
  double phi = a/std::sqrt(8*nu);
  double phi2 = phi*phi;
  double c2 = 1/(4*std::pow(std::tgamma(0.5)*phi,2)*nu);
  
  int n = nodeMat.rows();
  Eigen::SparseMatrix<double> I(n, n);
  I.setIdentity();
  
  Eigen::SparseMatrix<double> S = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["Shift"]);
  Eigen::SparseMatrix<double> Ch = Rcpp::as<Eigen::SparseMatrix<double> >(FEMatList["ScaleInv"]);
  Eigen::SparseMatrix<double> Q = c2*Ch*(I + phi2*S)*(I + phi2*S)*Ch;
  Eigen::SparseMatrix<double> Qsum = Q + (1/tau2)*(A.transpose())*A;
  
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > cholQ(Q);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > cholQsum(Qsum);
  
  
  Eigen::VectorXd Zk=KrigBase(cholQsum,Y,A, tau2);
  
  return CsampleBase(Zk,
                     cholQ,
                     cholQsum,
                     Y,  A,  tau2,
                     W, eps);
  
}



//-----------------------------------------------------------
//-----------------------------------------------------------

