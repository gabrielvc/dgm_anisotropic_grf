#include <Rcpp.h>
#include <RcppEigen.h>
#include <unsupported/Eigen/FFT>

#ifndef CHEBCOEFFSFFT
#define CHEBCOEFFSFFT


// Function objects:
class Funct{
protected:
  double d; // scaling factor
public:
  Funct() :  d(1.0) {}
  Funct(double d_) :  d(d_) {}
  ~Funct() {}
  virtual double operator()(double x) { return 0.0; }
  void scale(double d_){
    d=d_;
  }
};


// Function objects created from a R defined function
class FunctFromR : public Funct{
private:
  const Rcpp::Function f;
public:
  FunctFromR(Rcpp::Function f_) : Funct(), f(f_){}
  double operator()(double x)
  {
    return Rcpp::as<double>(f(x));
  }
};

// Function objects created from a C++ defined function
class FunctFromC : public Funct{
private:
  double (*f)(double);
public:
  FunctFromC(double (*f_)(double)) : Funct(), f(f_){}
  double operator()(double x)
  {
    return f(x);
  }
};




Eigen::VectorXd applyFunct(Funct* , Eigen::VectorXd );
Eigen::VectorXd coefsChebApprox(Funct*, int, double, double, int=1024 , int=0 );
Eigen::VectorXd coefsChebApprox(double (*)(double), int , double , double , int=0 );
Eigen::VectorXd coefsChebApprox(Rcpp::Function, int, double, double, int );

#endif
