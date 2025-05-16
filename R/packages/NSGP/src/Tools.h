#include <RcppEigen.h>


Eigen::SparseMatrix<double> SparseId(int);
double vecmin(Rcpp::NumericVector);
std::vector<int> argSort(const Eigen::ArrayXd& );
double splineRBF(double , int);
double DsplineRBF(double , int);
Eigen::ArrayXd lgt(Eigen::ArrayXd& ,double , double );
Eigen::ArrayXd lgtInv(Eigen::ArrayXd& ,double , double );
double lgt(double ,double , double );
