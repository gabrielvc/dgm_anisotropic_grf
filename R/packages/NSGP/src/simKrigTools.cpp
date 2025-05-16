// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>

//-----------------------------------------------------------

Eigen::MatrixXd NCsampleBase(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQ,
                             Eigen::MatrixXd& W){
  
  Eigen::ArrayXd vectDmh=cholQ.vectorD().array().sqrt().inverse();
  Eigen::MatrixXd res(W.rows(),W.cols());
  for(int i=0; i<W.cols(); ++i){
    res.col(i)= cholQ.permutationPinv()*(cholQ.matrixU()).solve((vectDmh* W.col(i).array()).matrix());
  }
  
  return res;
  
}


Eigen::VectorXd KrigBase(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQsum,
                         Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A, double tau2){
  
  
  Eigen::VectorXd Ytilde = (A.transpose())*Y;
  Eigen::VectorXd solveY = cholQsum.solve(Ytilde);
  
  return (1.0/tau2)*solveY;
  
}

Eigen::MatrixXd CsampleBase(Eigen::VectorXd& Zk,
                            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQ,
                            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& cholQsum,
                            Eigen::VectorXd& Y, Eigen::SparseMatrix<double>& A, double tau2,
                            Eigen::MatrixXd& W, Eigen::MatrixXd& eps){
  
  // Eigen::VectorXd Zk=KrigBase(cholQsum,Y,A, tau2);
  
  Eigen::MatrixXd Znc = NCsampleBase(cholQ, W);
  Eigen::MatrixXd Znck(Znc.rows(),Znc.cols());
  for(int i=0; i<W.cols(); ++i){
    Eigen::VectorXd Ync = A * Znc.col(i) + std::sqrt(tau2)*eps.col(i);
    Znck.col(i)=Zk + Znc.col(i) - KrigBase(cholQsum, Ync, A, tau2);
  }
  
  return Znck;
  
}
