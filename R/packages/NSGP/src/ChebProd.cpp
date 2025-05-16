// [[Rcpp::depends(RcppEigen)]]

#include <Eigen/SparseCholesky>

#include "ChebCoeffsFFT.h"
#include "Chebychev.h"
#include "Tools.h"


using namespace Rcpp;

typedef Eigen::Triplet<double> T;


//' Product of a Chebyshev matrix polynomial and  vectors
//'
//' \code{prodChebDecompVec} computes the product of a Chebyshev decomposition evaluated at a matrix and vectors binded in a matrix.
//'
//'@param S A sparse matrix.
//'@param Coefs A numeric vector. Coeffcients of a Chebyshev decomposition.
//'@param a,b Real numbers. Interval on which the Chebyshev polynomials are shifted. Default : \code{a=-1.0}, \code{b=1.0}.
//'@param V A matrix.
//'
//'@details This function computes the product
//'\deqn{\sum_{i=0}^{M-1} \mathtt{Coefs}[i+1]T_i(\frac{2}{b-a}S-\frac{a+b}{b-a}I)V}
//'where \code{M=length(Coefs)}, \eqn{I} is trhe identity matrix, \eqn{T_i} is the i-th Chebyshev poynomial.
//'
//'@return A matrix such that each column is the product of the evaluation at the matrix \code{S} of the Chebyshev decomposition, and a column of \code{V}. The coefficients of the decompostion are defined in \code{Coefs}.
//'
//'
// [[Rcpp::export]]
Eigen::MatrixXd prodChebDecompVec(Eigen::SparseMatrix<double> S, Eigen::VectorXd Coefs, double a, double b, Eigen::MatrixXd V){
  int Nc=Coefs.size();
  if(S.cols()!=S.rows()){
    Rcpp::Rcout<<"The matrix must be square!";
    return V;
  }
  if(S.cols()!=V.rows()){
    Rcpp::Rcout<<"Wrong dimension for matrix/vector product!";
    return V;
  }

  Eigen::MatrixXd M=Eigen::MatrixXd::Zero(S.rows(),V.cols()); // Contient l'approximation

  Eigen::MatrixXd v0(S.rows(),V.cols()), v1(S.rows(),V.cols()), v2(S.rows(),V.cols());// Storage
  Eigen::MatrixXd *Ch, *Chm1, *Chm2, *Chtemp;// Polynome de Cheb de l'iteration n-2
  Ch=&v0;
  Chm1=&v1;
  Chm2=&v2;

  double c1=4.0/(b-a), c2=2.0*((b+a)/(b-a));

  for(int n=0; n<Nc;++n){
    if(n==0){
      v2=V;
      M.noalias()+=Coefs(n)*v2; // mise à jour de l'approx
    }else if(n==1){
      v1=0.5*(c1*(S*V)-c2*V);
      M.noalias()+=Coefs(n)*v1; // mise à jour de l'approx
    }else{

      (*Ch)=c1*(S*(*Chm1))-c2*(*Chm1)-(*Chm2);
      M.noalias()+=Coefs(n)*(*Ch);

      Chtemp=Chm2;
      Chm2=Chm1;
      Chm1=Ch;
      Ch=Chtemp;

    }
  }
  return(M);
}



