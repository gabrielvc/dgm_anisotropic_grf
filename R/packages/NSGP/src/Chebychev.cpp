// [[Rcpp::depends(RcppEigen)]]

#include "Chebychev.h"
#include "Tools.h"


using namespace Rcpp;
typedef Eigen::Triplet<double> T;


//' Chebyshev Polynomial
//'
//' \code{evalCheb} returns the value of a Chebyshev polynomial evaluated at one (or several) point(s).
//'
//'@param x A real number or a vector.
//'@param n An integer. Order of the Chebyshev polynomial.
//'@param a,b Real numbers. Interval on which the Chebyshev polynomial is shifted. Default : \code{a=-1.0}, \code{b=1.0}.
//'
//'@return The evaluation at \code{x} of the \code{n}-th Chebyshev polynomial, shifted to interval \code{[a,b]}.
//'
//'@details This function computes
//'\eqn{T_n(\frac{2}{b-a}x-\frac{a+b}{b-a})}
//'where \eqn{T_n} is the n-th Chebyshev poynomial.
//'
//'
//'@examples
//'evalCheb(x=runif(10,-2,2), n=10, a=-2, b=2)
//'
// [[Rcpp::export]]
Eigen::VectorXd evalCheb(Eigen::VectorXd x, int n,double a=-1.0, double b=1.0){
  Eigen::VectorXd X(x);
  Chebyshev f(n,a,b);
  for(int i=0;i<X.size();++i){
    X(i)=f(x(i));
  }
  return X;
}
double evalCheb(double x, int n,double a=-1.0, double b=1.0){
  Chebyshev f(n,a,b);
  return f(x);
}


//' Evaluation of a Chebyshev decomposition
//'
//' \code{evalChebDecompPts} returns the value of a Chebyshev decomposition evaluated at several points.
//'
//'@param x A numeric vector.
//'@param Coefs A numeric vector. Coeffcients of a Chebyshev decomposition.
//'@param a,b Real numbers. Interval on which the Chebyshev polynomials are shifted. Default : \code{a=-1.0}, \code{b=1.0}.
//'
//'@details This function computes
//'\deqn{\sum_{i=0}^{M-1} \mathtt{Coefs}[i+1]T_i(\frac{2}{b-a}x-\frac{a+b}{b-a})}
//'where \code{M=length(Coefs)}, \eqn{T_i} is the i-th Chebyshev poynomial.
//'
//'@return The evaluation at each component of \code{x} of the Chebyshev decomposition whose coefficients are defined in \code{Coefs}.
//'
//'@examples
//'curve(evalChebDecompPts(x=x,Coefs=runif(5),a=-2,b=2),-2,2)
//'
// [[Rcpp::export]]
Eigen::ArrayXd evalChebDecompPts(Eigen::ArrayXd x, Eigen::ArrayXd Coefs, double a, double b){
  int N=Coefs.size();
  Eigen::ArrayXd Id = Eigen::ArrayXd::Ones(x.size());
  Eigen::ArrayXd M = Eigen::ArrayXd::Zero(x.size()); // Contient l'approximation

  Eigen::ArrayXd Ch(x.size()); // Polynome de Cheb n (iteration courante)
  Eigen::ArrayXd Chm1(x.size()); // Polynome de Cheb de l'iteration n-1
  Eigen::ArrayXd Chm2(x.size());// Polynome de Cheb de l'iteration n-2

  Eigen::ArrayXd Ch1=(2.0/(b-a))*x-((b+a)/(b-a))*Id; // Polynome de Cheb de l'iteration 1

  for(int n=0; n<N;++n){
    if(n==0){
      M=M+Coefs(n)*Id; // mise à jour de l'approx
    }else if(n==1){
      M=M+Coefs(n)*Ch1; // mise à jour de l'approx
      Chm2=Id;
      Chm1=Ch1;
    }else{
      Ch=2.0*Ch1*Chm1-Chm2; //Creation du polynome de Cheb de l'iterartion courante n
      M=M+Coefs(n)*Ch; // mise à jour de l'approx
      Chm2=Chm1; // Prerapration de l'iteration suivante : m1 -> m2 et courant -> m1
      Chm1=Ch;
    }
  }
  return(M);
}


//' Evaluation of a Chebyshev decomposition
//'
//' \code{evalChebDecompMat} returns the value of a Chebyshev decomposition evaluated at a matrix.
//'
//'@param S A sparse matrix.
//'@param Coefs A numeric vector. Coeffcients of a Chebyshev decomposition.
//'@param a,b Real numbers. Interval on which the Chebyshev polynomials are shifted. Default : \code{a=-1.0}, \code{b=1.0}.
//'
//'@details This function computes
//'\deqn{\sum_{i=0}^{M-1} \mathtt{Coefs}[i+1]T_i(\frac{2}{b-a}S-\frac{a+b}{b-a}I)}
//'where \code{M=length(Coefs)}, \eqn{I} is the indentity matrix, \eqn{T_i} is the i-th Chebyshev poynomial.
//'
//'
//'@return The evaluation at each component at the matrix \code{S} of the Chebyshev decomposition whose coefficients are defined in \code{Coefs}.
//'
// [[Rcpp::export]]
Eigen::SparseMatrix<double> evalChebDecompMat(Eigen::SparseMatrix<double> S, Eigen::VectorXd Coefs, double a, double b){
  int N=Coefs.size();
  Eigen::SparseMatrix<double> Id = SparseId(S.cols());

  if(S.cols()!=S.rows()){
    Rcpp::Rcout<<"The matrix must be square! The function will return the Identity matrix.";
    return Id;
  }

  Eigen::SparseMatrix<double> M(S.rows(),S.cols()); // Contient l'approximation
  Eigen::SparseMatrix<double> Ch(S.rows(),S.cols()); // Polynome de Cheb n (iteration courante)
  Eigen::SparseMatrix<double> Chm1(S.rows(),S.cols()); // Polynome de Cheb de l'iteration n-1
  Eigen::SparseMatrix<double> Chm2(S.rows(),S.cols());// Polynome de Cheb de l'iteration n-2

  Eigen::SparseMatrix<double> Ch1=(2.0/(b-a))*S-((b+a)/(b-a))*Id; // Polynome de Cheb de l'iteration 1

  for(int n=0; n<N;++n){
    if(n==0){
      M=M+Coefs(n)*Id; // mise à jour de l'approx
    }else if(n==1){
      M=M+Coefs(n)*Ch1; // mise à jour de l'approx
      Chm2=Id;
      Chm1=Ch1;
    }else{
      Ch=2.0*Ch1*Chm1-Chm2; //Creation du polynome de Cheb de l'iterartion courante n
      M=M+Coefs(n)*Ch; // mise à jour de l'approx
      Chm2=Chm1; // Prerapration de l'iteration suivante : m1 -> m2 et courant -> m1
      Chm1=Ch;
    }
  }

  return(M);
}


