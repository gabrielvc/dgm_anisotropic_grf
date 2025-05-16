
#include <Rcpp.h>
#include <RcppEigen.h>

#include "ChebCoeffsFFT.h"


#ifndef CHEBDEV
#define CHEBDEV


/*
 * Class whose objects are Chebyshev polynomials
 */
class Chebyshev: public Funct{
private:
  const int n; // Order of the polynomial
  const double a; // [a,b] : interval on which the polynomial is defined
  const double b;
public:
  Chebyshev(int n_, double a_, double b_) : n(n_), a(a_), b(b_) {}
  // Redefine operator () to return value of the polynomial at a point x
  double operator()(const double& x) const
  {
    if(n<=0){
      return 1;
    }else if(n==1){
      return (2.0/(b-a))*(x-a)-1;
    }else{
      return 2.0*((2.0/(b-a))*(x-a)-1)*Chebyshev(n-1,a,b)(x)-Chebyshev(n-2,a,b)(x);
    }
  }
};




Eigen::SparseMatrix<double> SparseId(int);
double evalCheb(double, int,double, double);
Eigen::VectorXd evalCheb(Eigen::VectorXd, int,double, double);
Eigen::ArrayXd evalChebDecompPts(Eigen::ArrayXd, Eigen::ArrayXd, double, double);
double evalChebDecompPts(double, Eigen::ArrayXd, double, double);
Eigen::SparseMatrix<double> evalChebDecompMat(Eigen::SparseMatrix<double>, Eigen::VectorXd, double, double);
Eigen::MatrixXd prodChebDecompMatVec(Eigen::SparseMatrix<double>, Eigen::VectorXd, double, double, Eigen::MatrixXd);

#endif
