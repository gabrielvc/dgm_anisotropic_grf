// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>

#include "ThinPlateSpline.h"



// [[Rcpp::export]]
Rcpp::List TPspline(Eigen::ArrayXXd knots, Eigen::ArrayXd val, int m, int p=2, double lambda=0){
  
  TPS T(knots,val,m,p,lambda);
  
  return Rcpp::List::create(Rcpp::_["c"]=T.get_c(), Rcpp::_["d"]=T.get_d(), Rcpp::_["knots"]=T.get_knots());
    
}


// [[Rcpp::export]]
Eigen::ArrayXd evalTPSpline(Eigen::ArrayXXd pts, Eigen::ArrayXXd knots, Eigen::ArrayXd val, int m, int p=2, double lambda=0){
  
  TPS T(knots,val,m,p,lambda);
  Eigen::ArrayXd res=T.eval(pts);
  
  return res;
  
}

// [[Rcpp::export]]
Eigen::ArrayXXd evalGradTPSpline(Eigen::ArrayXXd pts, Eigen::ArrayXXd knots, Eigen::ArrayXd val, int m, int p=2, double lambda=0){
  
  TPS T(knots,val,m,p,lambda);
  Eigen::ArrayXXd res=T.eval_grad(pts);
  
  return res;
  
}

// [[Rcpp::export]]
Eigen::ArrayXXd matCovarTPS(Eigen::ArrayXXd& xcoord, Eigen::ArrayXXd& knots, int m=3, int p=2){
  
  int npts = xcoord.rows();
  int nknots = knots.rows();
  Eigen::ArrayXXd res(npts,m*(m+1)/2+nknots);
  
  for(int k=0; k<npts; ++k){
    
    double x=xcoord(k,0);
    double y=xcoord(k,1);
    int cnt=0;
    for(int n=0; n<m;++n){
      for(int iy =0; iy< n+1; ++iy){
        res(k,cnt)=std::pow(x,n-iy)*std::pow(y,iy);
        cnt=cnt+1;
      }
    }
    for(int i=0; i< nknots; ++i){
      res(k,cnt)=splineRBF(std::pow(x-knots(i,0),2)+std::pow(y-knots(i,1),2),p);
      cnt=cnt+1;
    }
  }
  return res;
}
  

// [[Rcpp::export]]
Eigen::MatrixXd getTPSvalMat(Eigen::ArrayXXd& knots, int m=3, int p=2, double lambda=0){
  
  // Matrix of the linear system
  int nknots=knots.rows();
  int ncovar = m*(m+1) / 2;
  Eigen::MatrixXd M=Eigen::MatrixXd::Zero(nknots+ncovar,nknots+ncovar);
  
  // Compute covariance block
  for(int i=0; i<nknots;++i){
    for(int j=0; j<i;++j){
      double r2=(knots.row(j)-knots.row(i)).pow(2).sum();
      M(i,j)=splineRBF(r2,p);
      M(j,i)=M(i,j);
    }
  }
  
  // Compute covariate blocks
  int cnt=0;
  for(int n=0; n<m;++n){
    for(int iy =0; iy< n+1; ++iy){
      for(int k=0; k< nknots; ++k){
        M(k,nknots+cnt)=std::pow(knots(k,0),n-iy)*std::pow(knots(k,1),iy);
        M(nknots+cnt,k)=M(k,nknots+cnt);
      }
      cnt=cnt+1;
    }
  }
  
  // Add regularization
  if(lambda > 0){
    for(int i=0; i<nknots;++i){
      M(i,i)=M(i,i)+lambda;
    }
  }
  
  return M.inverse().matrix();
  
}



