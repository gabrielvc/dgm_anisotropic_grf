// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

using namespace Rcpp;

typedef Eigen::Triplet<double> T;

/*
 * Compute identity matrix
 */
Eigen::SparseMatrix<double> SparseId(int n){
  std::vector<T> tripletList;
  tripletList.reserve(n);
  for(int i=0; i<n;++i){
    tripletList.push_back(T(i,i,1));
  }
  Eigen::SparseMatrix<double> mat(n,n);
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
  return mat;
}

/*
 * Compute the minimum value of a NumericVector
 */
double vecmin(NumericVector x) {
  NumericVector::iterator it = std::min_element(x.begin(), x.end());
  return *it;
}


// [[Rcpp::export]]
std::vector<int> argSort(const Eigen::ArrayXd& array) {
  // Create a vector of indices
  std::vector<int> indices(array.size());
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }
  
  // Sort the indices based on the values in the Eigen array
  std::sort(indices.begin(), indices.end(),
            [&array](int a, int b) { return array[a] < array[b]; });
  
  return indices;
}

// RBF for splines
double splineRBF(double r2, int p){
  if(r2==0){
    return 0;
  }
  return std::pow(-1,p)*std::pow(r2,p-1)*std::log(r2)/(std::pow(2,2*p)*std::pow(std::tgamma(p)*std::tgamma(0.5),2));
}
// Derivative of RBF for splines
double DsplineRBF(double r2, int p){
  if(r2==0){
    return 0;
  }
  return std::pow(-1,p)*std::pow(r2,p-2)*((p-1)*std::log(r2)+1)/(std::pow(2,2*p)*std::pow(std::tgamma(p)*std::tgamma(0.5),2));
}


// Logit/Signoid functions
// [[Rcpp::export]]
Eigen::ArrayXd lgt(Eigen::ArrayXd& x,double rho_min=0, double rho_max=1){
  return rho_min+(rho_max-rho_min)/(1+(-x).exp());
}
// [[Rcpp::export]]
Eigen::ArrayXd lgtInv(Eigen::ArrayXd& x,double rho_min=0, double rho_max=1){
  return ((x-rho_min)/(rho_max-x)).log();
}
double lgt(double x,double rho_min=0, double rho_max=1){
  return rho_min+(rho_max-rho_min)/(1+std::exp(-x));
}






