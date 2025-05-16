// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include <Eigen/Dense>

#include "ChebCoeffsFFT.h"
#include "FemMat.h"


// [[Rcpp::export]]
Eigen::ArrayXd maternCov(Eigen::ArrayXd theta,Eigen::ArrayXd rho){
  
  double nu=theta(1);
  double a= theta(0);
  double kappa = std::sqrt(8*nu)/a;
  double pi=std::atan(1.0)*4;
  double tau2 = nu/(std::pow(2,nu-1)*std::tgamma(nu+1));
  Eigen::ArrayXd res(rho.size());
  for(int i=0; i<res.size(); ++i){
    if(rho(i)==0){
      res(i)= tau2*std::tgamma(nu)*std::pow(2,nu-1);
    }else{
      res(i)=tau2*std::cyl_bessel_k(nu,rho(i)*kappa)*std::pow(rho(i)*kappa,nu);
      // res(i)=std::exp(-std::sqrt(3)*rho(i)*kappa)*(1+std::sqrt(3)*rho(i)*kappa);
    }
  }
 return res;
}

double maternCov(Eigen::ArrayXd theta,Eigen::ArrayXd x1,Eigen::ArrayXd x2){
  
  double nu=theta(1);
  double a= theta(0);
  double kappa = std::sqrt(8*nu)/a;
  double pi=std::atan(1.0)*4;
  double tau2 = nu/(std::pow(2,nu-1)*std::tgamma(nu+1));
  double rho = std::pow((x1-x2).square().sum(),0.5);
  if(rho==0){
    return tau2*std::tgamma(nu)*std::pow(2,nu-1);
  }
  return tau2*std::cyl_bessel_k(nu,rho*kappa)*std::pow(rho*kappa,nu);
}

// [[Rcpp::export]]
Eigen::MatrixXd covMatMatern(Eigen::ArrayXd& theta, Eigen::ArrayXXd& Coord1, Eigen::ArrayXXd& Coord2){
  Eigen::MatrixXd cmat(Coord1.rows(),Coord2.rows());
  for(int i=0; i<cmat.rows(); ++i){
    for(int j=0; j< cmat.cols(); ++j){
      //cmat(i,j)=1;
      cmat(i,j)=maternCov(theta,Coord1.row(i),Coord2.row(j));
    }
  }
  return cmat;
}

Eigen::MatrixXd covMatMatern(Eigen::ArrayXd& theta, Eigen::ArrayXXd& obsCoord){
  
  Eigen::MatrixXd cmat(obsCoord.rows(),obsCoord.rows());
  for(int i=0; i<obsCoord.rows(); ++i){
    cmat(i,i)=maternCov(theta,obsCoord.row(i),obsCoord.row(i));
    for(int j=0;j< i; ++j){
      cmat(i,j)=maternCov(theta,obsCoord.row(i),obsCoord.row(j));
      cmat(j,i)=cmat(i,j);
    }
  }
  
  return cmat;
}


// // [[Rcpp::export]]
// double loglikStat(Eigen::ArrayXd theta, Eigen::VectorXd& Yobs,
//                   Eigen::ArrayXXd obsCoord, double tau2=0){
//   
//   Eigen::MatrixXd cmat = covMatMatern(theta,  obsCoord);
//   if(tau2>0){
//     for(int i=0;i<cmat.rows();++i){
//       cmat(i,i)+=tau2;
//     }
//   }
//   
//   Eigen::LDLT<Eigen::MatrixXd > ldlt(cmat);
//   
//   return  -0.5*ldlt.vectorD().log().sum()-0.5*((ldlt.solve(Yobs).array())*(Yobs.array())).sum();
//   
// }

// [[Rcpp::export]]
double loglikStat(Eigen::ArrayXd& theta, Eigen::VectorXd& Yobs,
                  Eigen::ArrayXXd& obsCoord, double tau2=0){
  
  Eigen::MatrixXd cmat = covMatMatern(theta,  obsCoord);
  if(tau2>0){
    for(int i=0; i<cmat.rows();++i){
      cmat(i,i)+=tau2;
    }
  }
  
  Eigen::LLT<Eigen::MatrixXd > ldlt(cmat);
  
  double res = 0;
  Eigen::MatrixXd L = ldlt.matrixL();
  for(int i=0; i< L.rows(); ++i){
    res=res-std::log(L(i,i));
  }
  res=res-0.5*((ldlt.solve(Yobs).array())*(Yobs.array())).sum();

  return  res;
  
}

//-----------------------------------------------------------



//-----------------------------------------------------------

//Kriging
Eigen::VectorXd KrigStatBase(Eigen::MatrixXd& cTarObs, Eigen::VectorXd& Yobs,
                             Eigen::LLT<Eigen::MatrixXd >& cholSObs){
  
  return cTarObs*(cholSObs.solve(Yobs));
  
}
// [[Rcpp::export]]
Eigen::VectorXd KrigStat(Eigen::ArrayXXd& tarCoord,
                         Eigen::ArrayXd& theta, Eigen::VectorXd& Yobs,
                         Eigen::ArrayXXd& obsCoord, double tau2=0){
  
  Eigen::MatrixXd cTarObs = covMatMatern(theta,  tarCoord,obsCoord);
  Eigen::MatrixXd cObsObs = covMatMatern(theta,  obsCoord);
  if(tau2>0){
    for(int i=0;i<cObsObs.rows();++i){
      cObsObs(i,i)+=tau2;
    }
  }
  Eigen::LLT<Eigen::MatrixXd > cholS(cObsObs);
  
  return KrigStatBase(cTarObs, Yobs, cholS);
  
}


// [[Rcpp::export]]
Eigen::VectorXd KrigGaniso(Eigen::ArrayXXd& tarCoord,
                         Eigen::ArrayXd& theta, Eigen::VectorXd& Yobs,
                         Eigen::ArrayXXd& obsCoord, double tau2=0){
  
  
  Eigen::Matrix2d rotMat;
  rotMat << std::cos(-theta(2)), -std::sin(-theta(2)),
            std::sin(-theta(2)), std::cos(-theta(2));
  Eigen::ArrayXXd obsCoordRot = (rotMat * (obsCoord.matrix().transpose())).transpose();
  Eigen::ArrayXXd tarCoordRot = (rotMat * (tarCoord.matrix().transpose())).transpose();
  obsCoordRot.col(1)=obsCoordRot.col(1)/theta(3);
  tarCoordRot.col(1)=tarCoordRot.col(1)/theta(3);
  
  Eigen::MatrixXd cTarObs = covMatMatern(theta,  tarCoordRot,obsCoordRot);
  Eigen::MatrixXd cObsObs = covMatMatern(theta,  obsCoordRot);
  if(tau2>0){
    for(int i=0;i<cObsObs.rows();++i){
      cObsObs(i,i)+=tau2;
    }
  }
  Eigen::LLT<Eigen::MatrixXd > cholS(cObsObs);
  
  return KrigStatBase(cTarObs, Yobs, cholS);
  
}


// Non conditional simulation
Eigen::MatrixXd NCondSampleStatBase(Eigen::LLT<Eigen::MatrixXd > cholS,
                                    Eigen::MatrixXd& W){
  Eigen::MatrixXd res(W.rows(),W.cols());
  for(int i=0; i<W.cols(); ++i){
    res.col(i)= (cholS.matrixL()*((W.col(i).array()).matrix()));
  }
  return res;
}
// [[Rcpp::export]]
Eigen::MatrixXd NCondSampleStat(Eigen::ArrayXXd& tarCoord,
                                 Eigen::ArrayXd& theta,
                                 Eigen::MatrixXd& W){
  
  Eigen::MatrixXd cmat = covMatMatern(theta,  tarCoord);
  Eigen::LLT<Eigen::MatrixXd > cholS(cmat);

  return NCondSampleStatBase(cholS,W);
}


// Conditional simulation
// [[Rcpp::export]]
Eigen::MatrixXd CondSampleStat(Eigen::ArrayXXd tarCoord,
                               Eigen::ArrayXd& theta, Eigen::VectorXd& Yobs,
                               Eigen::ArrayXXd obsCoord,
                               Eigen::MatrixXd& W,Eigen::MatrixXd& eps,
                               double tau2=0){
  
  
  Eigen::MatrixXd cTarObs = covMatMatern(theta,  tarCoord,obsCoord);
  Eigen::MatrixXd cObsObs = covMatMatern(theta,  obsCoord);
  if(tau2>0){
    for(int i=0;i<cObsObs.rows();++i){
      cObsObs(i,i)+=tau2;
    }
  }
  Eigen::LLT<Eigen::MatrixXd > cholSObs(cObsObs);
  Eigen::VectorXd Zk = KrigStatBase(cTarObs, Yobs, cholSObs);
  
  int ntar=tarCoord.rows();
  int nobs=obsCoord.rows();
  Eigen::ArrayXXd allCoord(ntar+nobs,tarCoord.cols());
  for(int i=0; i<ntar;++i){
    allCoord.row(i)=tarCoord.row(i);
  }
  for(int i=0; i<nobs;++i){
    allCoord.row(ntar+i)=obsCoord.row(i);
  }
  Eigen::MatrixXd cmat = covMatMatern(theta,  allCoord);
  Eigen::LLT<Eigen::MatrixXd > cholS(cmat);
  Eigen::MatrixXd Znc = NCondSampleStatBase(cholS, W);
  
  
  Eigen::MatrixXd Znck(ntar,Znc.cols());
  Eigen::VectorXd YnewObs(ntar), ZnewTar(ntar);
  for(int i=0; i<W.cols(); ++i){
    for(int k=0; k<ntar;++k){
      ZnewTar(k)=Znc(k,i);
    }
    for(int k=0; k<nobs;++k){
      YnewObs(k)=Znc(ntar+k,i)+std::sqrt(tau2)*eps(k,i);
    }
    
    Znck.col(i)=Zk + ZnewTar - KrigStatBase(cTarObs, YnewObs, cholSObs);
  }
  
  return Znck;
  
  
}

//------------------------------------------------------------------






