#include <Rcpp.h>
#include <RcppEigen.h>

#ifndef CHEBPROD
#define CHEBPROD

Eigen::MatrixXd prodChebDecompVec(Eigen::SparseMatrix<double> , Eigen::VectorXd , double , double , Eigen::MatrixXd );

#endif
