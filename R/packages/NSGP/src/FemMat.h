#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Sparse>



#ifndef FEMDEF
#define FEMDEF

Rcpp::List matFEM2d(Eigen::ArrayXXd& , Eigen::ArrayXXi& ,
                    Eigen::ArrayXXd& ,
                    Rcpp::Nullable<Rcpp::NumericVector > = R_NilValue,
                    Rcpp::Nullable<Rcpp::NumericMatrix > = R_NilValue,
                    bool = true);

#endif
