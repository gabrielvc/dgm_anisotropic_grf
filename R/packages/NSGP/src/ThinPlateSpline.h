
#include <Rcpp.h>
#include <RcppEigen.h>
#include "Tools.h"


#ifndef TPS_DEF
#define TPS_DEF



/*
 * Class whose objects are Chebyshev polynomials
 */
class TPS{
private:
  const Eigen::ArrayXXd knots; // Order of the polynomial
  const int m; // [a,b] : interval on which the polynomial is defined
  const int p;
  const double lambda;
  Eigen::VectorXd c;
  Eigen::VectorXd d;
  
public:
  TPS(Eigen::ArrayXXd knots_, Eigen::VectorXd c_, Eigen::VectorXd d_, int m_, int p_, double lambda_) : knots(knots_), c(c_), d(d_), m(m_), p(p_), lambda(lambda_) {}
  TPS(Eigen::ArrayXXd knots_, Eigen::ArrayXd val_, int m_, int p_, double lambda_) : knots(knots_), m(m_), p(p_), lambda(lambda_) {
    
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
    
    // Form RHS
    Eigen::VectorXd b =  Eigen::VectorXd::Zero(nknots+ncovar);
    for(int i=0; i<nknots;++i){
      b(i)=val_(i);
    } 
    
    //Solve linear system
    Eigen::VectorXd sol= M.colPivHouseholderQr().solve(b);
    
    // Export
    c=Eigen::VectorXd::Zero(nknots);
    for(int i=0; i<nknots;++i){
      c(i)=sol(i);
    } 
    d=Eigen::VectorXd::Zero(ncovar);
    for(int i=0; i<ncovar;++i){
      d(i)=sol(nknots+i);
    } 
    
  }
  
  ~TPS() {}
  
  double eval(double x, double y){
    
    double s=0;
    
    int cnt=0;
    for(int n=0; n<m;++n){
      for(int iy =0; iy< n+1; ++iy){
        s=s+d(cnt)*std::pow(x,n-iy)*std::pow(y,iy);
        cnt=cnt+1;
      }
    }
    for(int i=0; i< c.size(); ++i){
      s=s+c(i)*splineRBF(std::pow(x-knots(i,0),2)+std::pow(y-knots(i,1),2),p);
    }
    return s;
  
  }
  
  Eigen::Vector2d eval_grad(double x, double y){
    
    Eigen::Vector2d g = Eigen::Vector2d::Zero();
    int cnt=0;
    for(int n=0; n<m;++n){
      for(int iy =0; iy< n+1; ++iy){
        if(n-iy > 0){
          g(0)=g(0)+d(cnt)*(n-iy)*std::pow(x,n-iy-1)*std::pow(y,iy);
        }
        if(iy > 0){
          g(1)=g(1)+d(cnt)*std::pow(x,n-iy)*iy*std::pow(y,iy-1);
        }
        cnt=cnt+1;
      }
    }
    for(int i=0; i< c.size(); ++i){
      g(0)=g(0)+2*c(i)*(x-knots(i,0))*DsplineRBF(std::pow(x-knots(i,0),2)+std::pow(y-knots(i,1),2),p);
      g(1)=g(1)+2*c(i)*(y-knots(i,1))*DsplineRBF(std::pow(x-knots(i,0),2)+std::pow(y-knots(i,1),2),p);
    }
    return g;
    
  }
  
  Eigen::ArrayXd eval(Eigen::ArrayXXd pts){
    
    int npts = pts.rows();
    Eigen::ArrayXd res = Eigen::ArrayXd::Zero(npts);
    for(int i=0; i< npts; ++i){
      res(i)=eval(pts(i,0),pts(i,1));
    }
    
    return res;
  }
  
  Eigen::ArrayXXd eval_grad(Eigen::ArrayXXd pts){
    
    int npts = pts.rows();
    Eigen::ArrayXXd res = Eigen::ArrayXXd::Zero(npts,2);
    for(int i=0; i< npts; ++i){
      res.row(i)=eval_grad(pts(i,0),pts(i,1));
    }
    return res;
  }
  
  Eigen::VectorXd get_c(){
    return c;
  }
  Eigen::VectorXd get_d(){
    return d;
  }
  Eigen::VectorXd get_knots(){
    return knots;
  }
  int get_m(){
    return m;
  }
  int get_p(){
    return p;
  }
  double get_lambda(){
    return lambda;
  }
  
};

#endif


