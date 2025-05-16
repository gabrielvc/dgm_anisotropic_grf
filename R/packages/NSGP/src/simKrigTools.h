#include <Rcpp.h>
#include <RcppEigen.h>

//-----------------------------------------------------------

Eigen::MatrixXd NCsampleBase(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQ,
                             Eigen::MatrixXd& W);


Eigen::VectorXd KrigBase(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQsum,
                         Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A, double tau2);

Eigen::MatrixXd CsampleBase(Eigen::VectorXd& Zk,
                            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQ,
                            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQsum,
                            Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A, double tau2,
                            Eigen::MatrixXd& W, Eigen::MatrixXd& eps);
