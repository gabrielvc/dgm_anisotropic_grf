
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include "ChebCoeffsFFT.h"
#include <unsupported/Eigen/FFT>

using namespace Rcpp;

//' Apply a function defined in R to a vector
//'
//'@param f Pointer to a Funct object.
//'@param x Numerical vector.
//'
//'@return The numerical vector f(x).
//'
//'@keywords internal
//'
Eigen::VectorXd applyFunct(Funct* f, Eigen::VectorXd x){
  int n=x.size();
  Eigen::VectorXd y(n);
  for(int i=0;i<n;++i){
    y(i)=(*f)(x(i));
  }
  return y;
}


//' Initialize vector for computation of coefficents of Chebyshev approximation by FFT
//'
//'@param f Pointer to a Funct object. Function to be approximated.
//'@param theta Numerical vector. Discretisation of the interval [0,2*pi].
//'@param a,b Real numbers. Interval on which the approximation is carried out.
//'
//'@return A matrix whose both columns are used to compute the coefficients using FFT.
//'
//'@keywords internal
//'
Eigen::MatrixXcd initFFT(Funct* f, Eigen::ArrayXd theta, double a, double b){
  int n=theta.size();
  Eigen::MatrixXcd y(n,2);
  double temp;
  for(int i=0;i<n;++i){
    y(i,0)=std::complex<double>(0.5*((*f)((a+b)*0.5+((b-a)*0.5)*cos(theta(i)*0.5))+(*f)((a+b)*0.5-((b-a)*0.5)*cos(theta(i)*0.5))),0.0);
    temp = 0.5*((*f)((a+b)*0.5+((b-a)*0.5)*cos(theta(i)*0.5))-(*f)((a+b)*0.5-((b-a)*0.5)*cos(theta(i)*0.5)));
    y(i,1)=std::complex<double>(temp*cos(theta(i)*0.5),-temp*sin(theta(i)*0.5));
  }
  return y;
}



// Initialize FFT
Eigen::FFT<double> fft;
// Apply FFT to a vector
Eigen::VectorXcd fftCpp(Eigen::VectorXcd x){
  return(fft.fwd(x));
}



//' Computation of coefficents of Chebyshev approximation by FFT
//'
//'@param h Pointer to a Funct object. Function to be approximated.
//'@param M Integer. Order of the polynomial approximation.
//'@param a,b Real numbers. Interval on which the approximation is carried out.
//'@param minSubdiv Integer. Minimal number of discretisation points.
//'@param verbose Integer. Print messages if \code{verbose=1}.
//'
//'@return A vector containing the \code{M+1} coefficients of the Chebyshev approximation of \code{h} over [\code{a},\code{b}].
//'
//'@keywords internal
//'
Eigen::VectorXd coefsChebApprox(Funct* h, int M, double a, double b, int minSubdiv, int verbose){
  double Pi =std::acos(-1);

  // Nombre de subdivisions
  int N=0;
  if(2*minSubdiv >= (M+1)){
    N=minSubdiv;
  }else{
    if(verbose==1){
      Rcout<<"Attention : l'agorithme sera plus optimisé si le paramètre minSubdiv est pris égal à une puissance de 2 supérieure à l'ordre souhaité";
    }
    N=((M+1)/2)+1;
  }

  Eigen::ArrayXd theta = (((double)(2.0*Pi))/N)*Eigen::ArrayXd::LinSpaced(N,0,N-1);
  Eigen::MatrixXcd MatFuncFFT = initFFT(h,theta,a,b);

  Eigen::VectorXd x =(2.0/N)*(fftCpp(MatFuncFFT.col(0)).real());  //Coefficients correspondant aux polynomes d'ordre pair
  Eigen::VectorXd y =(2.0/N)*(fftCpp(MatFuncFFT.col(1)).real());  //Coefficients correspondant aux polynomes d'ordre pair
  x(0)=0.5*x(0);  //Le coefficient d'ordre 0 a un facteur de normalisation différent

  Eigen::VectorXd FFTCoefs(M+1);
  for(int m=0;m<(M+1);++m){
    if(m%2 == 1){
      FFTCoefs(m)=y((m-1)/2);
    }else{
      FFTCoefs(m)=x(m/2);
    }
  }
  return(FFTCoefs);
}

// Export for use in R
//'Coefficents of Chebyshev approximation by FFT
//'
//' Computation of the coefficents of Chebyshev approximation of a function over a given interval (by FFT).
//'
//'@param h R numerical function. Function to be approximated.
//'@param M Integer. Order of the polynomial approximation.
//'@param a,b Real numbers. Interval on which the approximation is carried out.
//'@param minSubdiv Integer. Minimal number of discretisation points.
//'@param verbose Integer. Print messages if \code{verbose=1}.
//'
//'@return A vector containing the \code{M+1} coefficients of the Chebyshev approximation of \code{h} over [\code{a},\code{b}].
//'
//'@examples
//'f<-function(x){
//' (1+x)**(-2)
//'}
//'coefsChebApprox(f,M=100,a=0,b=50)
//'
// [[Rcpp::export]]
Eigen::VectorXd coefsChebApprox(Rcpp::Function h_, int M, double a, double b, int verbose=0){
  int minSubdiv = std::pow(2,std::max(10.0,1.0+std::ceil(std::log(std::max(1,M))/std::log(2))));
  if(verbose==1){
    Rcpp::Rcout<<"Computing "<<M<<" coefficients, using a discetization size of "<<minSubdiv<<".";
  }
  FunctFromR h(h_);
  Funct* hp=&h;
  Eigen::VectorXd res=coefsChebApprox(hp,M,a,b,minSubdiv,verbose);

  return(res);
}

// Same as above for for use in C++
Eigen::VectorXd coefsChebApprox(double (*h_)(double), int M, double a, double b, int verbose){
  int minSubdiv = std::pow(2,std::max(10.0,1.0+std::ceil(std::log(std::max(1,M))/std::log(2))));
  FunctFromC h(h_);
  Funct* hp=&h;
  Eigen::VectorXd  res=coefsChebApprox(hp,M,a,b,minSubdiv,verbose);
  return(res);
}

