// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include <Eigen/Dense>
#include <random>

#include "logLikStat.h"


// [[Rcpp::export]]
double loglikGAniso(Eigen::ArrayXd& theta, Eigen::VectorXd& Yobs,
                  Eigen::ArrayXXd& obsCoord, double tau2=0){
  
  Eigen::Matrix2d rotMat;
  rotMat << std::cos(-theta(2)), -std::sin(-theta(2)),
            std::sin(-theta(2)), std::cos(-theta(2));
  Eigen::ArrayXXd obsCoordRot = (rotMat * (obsCoord.matrix().transpose())).transpose();
  obsCoordRot.col(1)=obsCoordRot.col(1)/theta(3);
  
  return loglikStat(theta,Yobs,obsCoordRot,  tau2);

}

//-----------------------------------------------------------

bool inRange(double x, double minx, double maxx){
  return ((x>= minx)&&(x<= maxx)) ;
}

// [[Rcpp::export]]
double logPostGAniso(Eigen::ArrayXd& theta, Eigen::VectorXd& Yobs,
                    Eigen::ArrayXXd& obsCoord, double tau2=0){
  

  // Log prior
  double lp =0;
  if( inRange(theta(0),0.05,0.3)  && inRange(theta(2),0,std::atan(1.0)*4)  && inRange(theta(3),0.1,1.0)){
    lp=1;
  }else{
    return - std::numeric_limits<double>::infinity();
  }
  
  // Log likelihood
  double ll = loglikGAniso(theta,Yobs,obsCoord,  tau2);
  
  return lp + ll;
}



//-----------------------------------------------------------

// [[Rcpp::export]]
Eigen::ArrayXXd RW_MCMC(Eigen::ArrayXd theta_init, Eigen::VectorXd& Yobs,
                       Eigen::ArrayXXd& obsCoord, double tau2=0, int nbIter=100, int Thinning = 10,
                       Eigen::ArrayXd stdVec=Eigen::ArrayXd::Ones(4)){
  
  // random device class instance
  std::random_device rd{}; 
  // Mersenne twister PRNG,
  std::mt19937 gen{rd()}; 
  std::normal_distribution rnorm{0.0, 1.0};
  std::uniform_real_distribution<>  runif(0.0, 1.0);
  
  
  int nbSaved = nbIter / Thinning;
  if(nbSaved * Thinning < nbIter){
    nbSaved +=2;
  }else{
    nbSaved+=1;
  }
  Eigen::ArrayXXd theta_saved = Eigen::ArrayXXd::Zero(nbSaved,theta_init.size());
  Eigen::ArrayXd theta = theta_init;
  Eigen::ArrayXd theta_prop = Eigen::ArrayXd::Zero(theta.size());
  
  //Itinitalize
  theta_saved.row(0)=theta;
  
  double lp_cur=0, lp_prop=0, mh=0;
  lp_cur=logPostGAniso(theta, Yobs,obsCoord, tau2);
  
  // Eigen::ArrayXd theta;
  int cnt_skip=0, cnt_save=0;
  for(int i =0; i<nbIter; ++i ){
    
    cnt_skip+=1;
    
    //Proposal
    for(int k=0; k< theta.size(); ++k){
      theta_prop(k)=theta(k)+stdVec(k)*rnorm(gen);
    } 
    lp_prop=logPostGAniso(theta_prop, Yobs,obsCoord, tau2);
    mh=std::exp(lp_prop-lp_cur);
    
    //Accept or reject
    if(runif(gen) < mh){
      theta=theta_prop;
      lp_cur=lp_prop;
    }
    
    if((cnt_skip==Thinning)||(i==(nbIter-1))){
      cnt_skip=0;
      cnt_save+=1;
      theta_saved.row(cnt_save)=theta;
    }
    
  }
  
  return theta_saved;
  
}

