#include <Rcpp.h>
#include <RcppEigen.h>

#ifndef FUNCT
#define FUNCT


// Function objects:
class FuncVect{
public:
  FuncVect(){}
  ~FuncVect() {}
  virtual Eigen::ArrayXd operator()(Eigen::ArrayXd x) { return Eigen::ArrayXd::Zero(1); }
};


// Function objects created from a R defined function
class FuncVectFromR : public FuncVect{
private:
  const Rcpp::Function f;
public:
  FuncVectFromR(Rcpp::Function f_) : FuncVect(), f(f_){}
  Eigen::ArrayXd operator()(Eigen::ArrayXd x)
  {
    return Rcpp::as<Eigen::ArrayXd >(f(x));
  }
};


#endif


