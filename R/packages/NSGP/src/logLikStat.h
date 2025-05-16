#include <RcppEigen.h>

Eigen::MatrixXd covMatMatern(Eigen::ArrayXd& , Eigen::ArrayXXd& , Eigen::ArrayXXd& );
Eigen::MatrixXd covMatMatern(Eigen::ArrayXd& , Eigen::ArrayXXd& );
double loglikStat(Eigen::ArrayXd&, Eigen::VectorXd&,Eigen::ArrayXXd&, double );
