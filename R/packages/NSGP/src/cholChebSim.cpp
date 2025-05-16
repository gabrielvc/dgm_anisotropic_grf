// [[Rcpp::depends(RcppEigen)]]

#include <Eigen/SparseCholesky>

#include "Chebychev.h"
//#include "tools.h"


using namespace Rcpp;


//product by ((2.0/(b-a))*S-((b+a)/(b-a))*Id)
Eigen::MatrixXd prodShiftChol(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& CholSolver, Eigen::SparseMatrix<double>& R, Eigen::ArrayXd& Ddm,  Eigen::MatrixXd& V, double& a, double& b){

  Eigen::VectorXd x(V.rows());
  Eigen::MatrixXd res(V.rows(),V.cols());

  for(int i=0; i<V.cols();++i){
    x=V.col(i);
    res.col(i)=-((b+a)/(b-a))*x;

    //Multplication by S
    x=((x.array())*Ddm).matrix();
    x=CholSolver.matrixU().solve(x);
    x=CholSolver.permutationPinv()*x;

    x=R*x;

    x=CholSolver.permutationP()*x;
    x=CholSolver.matrixL().solve(x);
    x=((x.array())*Ddm).matrix();

    res.col(i)+=(2.0/(b-a))*x;
  }

  return res;
}


Eigen::MatrixXd prodShiftCholMat(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> >& CholSolver, Eigen::SparseMatrix<double>& R, Eigen::ArrayXd& Ddm,  Eigen::MatrixXd& V, double& a, double& b){


  Eigen::MatrixXd x=V;

  Eigen::MatrixXd res =-((b+a)/(b-a))*x;

  //Multplication by S
  for(int i =0; i<x.cols(); ++i){
    x.col(i)=((x.col(i).array())*Ddm).matrix();
  }
  x=CholSolver.matrixU().solve(x);
  x=CholSolver.permutationPinv()*x;

  x=R*x;

  x=CholSolver.permutationP()*x;
  x=CholSolver.matrixL().solve(x);
  for(int i =0; i<x.cols(); ++i){
    x.col(i)=((x.col(i).array())*Ddm).matrix();
  }

  res+=(2.0/(b-a))*x;


  return res;
}


// [[Rcpp::export]]
Eigen::MatrixXd simChebChol(Eigen::SparseMatrix<double> C, Eigen::SparseMatrix<double> R, Eigen::VectorXd Coefs, double a, double b, Eigen::MatrixXd V){

  int Nc=Coefs.size();

  if(R.cols()!=R.rows()){
    Rcpp::Rcout<<"The matrix must be square!";
    return V;
  }
  if(R.cols()!=V.rows()){
    Rcpp::Rcout<<"Wrong dimension for matrix/vector product!";
    return V;
  }

  Eigen::MatrixXd M=Eigen::MatrixXd::Zero(R.rows(),V.cols()); // Contient l'approximation

  Eigen::MatrixXd Ch(R.rows(),V.cols()); // Polynome de Cheb n (iteration courante)
  Eigen::MatrixXd Chm1(R.rows(),V.cols()); // Polynome de Cheb de l'iteration n-1
  Eigen::MatrixXd Chm2(R.rows(),V.cols());// Polynome de Cheb de l'iteration n-2


  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > CholSolver(C);
  Eigen::ArrayXd Ddm=1.0/(CholSolver.vectorD().array().sqrt());

  Eigen::MatrixXd Ch1=prodShiftCholMat(CholSolver, R, Ddm, V, a, b); // Polynome de Cheb de l'iteration 1

  for(int n=0; n<Nc;++n){
    if(n==0){
      M=M+Coefs(n)*V; // mise à jour de l'approx
    }else if(n==1){
      M=M+Coefs(n)*Ch1; // mise à jour de l'approx
      Chm2=V;
      Chm1=Ch1;
    }else{
      Ch.noalias()=2.0*(prodShiftCholMat(CholSolver, R,Ddm, Chm1, a, b))-Chm2; //Creation du polynome de Cheb de l'iterartion courante n
      M.noalias()+=Coefs(n)*Ch; // mise à jour de l'approx
      Chm2=Chm1; // Prerapration de l'iteration suivante : m1 -> m2 et courant -> m1
      Chm1=Ch;
    }
  }

  Eigen::VectorXd x(M.rows());
  for(int i=0; i<V.cols();++i){
    x=((M.col(i).array())*Ddm).matrix();
    x=CholSolver.matrixU().solve(x);
    x=CholSolver.permutationPinv()*x;
    M.col(i)=x;
  }

  return(M);
}



Eigen::MatrixXd applyCmd(Eigen::SparseMatrix<double> C, Eigen::SparseMatrix<double> R, Eigen::MatrixXd M){

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > CholSolver(C);
  Eigen::ArrayXd Ddm=1.0/(CholSolver.vectorD().array().sqrt());

  Eigen::VectorXd x(M.rows());
  for(int i=0; i<M.cols();++i){
    x=CholSolver.permutationP()*(M.col(i));
    x=((x.array())*Ddm).matrix();
    x=CholSolver.matrixU().solve(x);
    x=CholSolver.permutationPinv()*x;
    M.col(i)=x;
  }

  return M;

}


// [[Rcpp::export]]
Eigen::MatrixXd projectNoise(Eigen::SparseMatrix<double> C_fine,Eigen::SparseMatrix<double> C_coarse, Eigen::SparseMatrix<double> Amat, Eigen::MatrixXd W){
  
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > CholSolver_fine(C_fine);
  Eigen::ArrayXd Dd_fine=(CholSolver_fine.vectorD().array().sqrt());

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > CholSolver_coarse(C_coarse);
  Eigen::ArrayXd Ddm_coarse=1.0/(CholSolver_coarse.vectorD().array().sqrt());
  
  Eigen::MatrixXd M=Eigen::MatrixXd::Zero(C_coarse.rows(),W.cols()); // Contient l'approximation
  
  
  Eigen::VectorXd x(C_fine.rows());
  Eigen::VectorXd y(C_coarse.rows());
  for(int i=0; i<W.cols();++i){
    x=((W.col(i).array())*Dd_fine).matrix();
    x=CholSolver_fine.matrixL()*x;
    x=CholSolver_fine.permutationPinv()*x;
    y=Amat*x;
    y=CholSolver_coarse.permutationP()*y;
    y=CholSolver_coarse.matrixL().solve(y);
    y=((y.array())*Ddm_coarse).matrix();
    M.col(i)=y;
  }
  
  return M;
  
}



double lambMinC(Eigen::SparseMatrix<double> C){

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > CholSolver(C);

  return CholSolver.vectorD().minCoeff();

}
