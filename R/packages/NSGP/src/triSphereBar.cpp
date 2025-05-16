// [[Rcpp::depends(RcppEigen)]]


// This file contains a function that returns the baycentric coordinates
// of points in a shphre wrt to a triangulation

#include <Rcpp.h>
#include <RcppEigen.h>
#include "Tools.h"

using namespace Rcpp;
// Compile with: gcc -std=c++11 -lm -O3 -ffast-math -o perftest perftest.cpp




Eigen::VectorXd crossp(Eigen::VectorXd u, Eigen::VectorXd v){

  Eigen::VectorXd res=Eigen::ArrayXd::Zero(3);
  res(0)=u(1) * v(2) - u(2) * v(1);
  res(1)=u(2) * v(0) - u(0) * v(2);
  res(2)=u(0) * v(1) - u(1) * v(0);
  return res;
}

/* Ray-triangle intersection routine */
// [[Rcpp::export]]
Eigen::VectorXd rayTriangleIntersect(Eigen::VectorXd dir, Eigen::VectorXd v0, Eigen::VectorXd v1, Eigen::VectorXd v2)
{

  Eigen::VectorXd res=-1.0+Eigen::ArrayXd::Zero(3);
  Eigen::VectorXd pvec = crossp(dir,v2-v0);
  double det =(pvec.dot(v1-v0));
  double eps=1e-10;
  if (std::abs(det)< eps){
    return res;
  }
  double invDet = 1.0 / det;
  double u = -pvec.dot(v0) * invDet;
  if (u < - eps || u > 1+eps){
    return res;
  }

  Eigen::VectorXd qvec = crossp(-v0,v1-v0);
  double v = dir.dot(qvec) * invDet;
  if (v < -eps || u + v > 1+eps){
    return res;
  }
  res(0)=qvec.dot(v2-v0) * invDet; 
  res(1)=u; // barycentric coordinate of v1
  res(2)=v; // barycentric coordinate of v2

  return res;
}



// [[Rcpp::export]]
Eigen::MatrixXd sphBarCoord(Eigen::MatrixXd& sphPts, Eigen::MatrixXd& nodeMat, Eigen::ArrayXXi& triMat){

  Eigen::MatrixXd res=Eigen::MatrixXd::Zero(sphPts.rows(),4);
  Eigen::VectorXd w;

  int iv0, iv1, iv2;
  for(int k=0; k<sphPts.rows(); ++k){
    bool notFound=true;
    int i=0;
    while((i<triMat.rows())&&(notFound)){
      iv0=triMat(i,0);
      iv1=triMat(i,1);
      iv2=triMat(i,2);
      w=rayTriangleIntersect(sphPts.row(k),nodeMat.row(iv0),nodeMat.row(iv1),nodeMat.row(iv2));
      if(w(0)>0){
        res(k,0)=i; // Index of the triangle 
        res(k,1)=w(0); 
        res(k,2)=w(1); // barycentric coordinate of iv1
        res(k,3)=w(2); // barycentric coordinate of iv2
        notFound=false;
      }
    i+=1;
    }
    
    // if(notFound){
    //     Rcpp::Rcout<<k<<" not found\n";
    // }

  }

  return res;

}



// [[Rcpp::export]]
Eigen::MatrixXd sphBarCoordFast(Eigen::MatrixXd& sphPts, Eigen::MatrixXd& nodeMat, Eigen::ArrayXXi& triMat,
                                Eigen::ArrayXXd& triBar){
  
  Eigen::MatrixXd res=Eigen::MatrixXd::Zero(sphPts.rows(),4);
  Eigen::VectorXd w;
  
  int iv0, iv1, iv2;
  Eigen::ArrayXd cur_pt=Eigen::ArrayXd::Zero(sphPts.cols());
  Eigen::ArrayXd tmp=Eigen::ArrayXd::Zero(triBar.rows());
  std::vector<int> it;
  
  for(int k=0; k<sphPts.rows(); ++k){
    bool notFound=true;
    int i0=0;
    cur_pt=sphPts.row(k);
    tmp=(triBar.col(0)-cur_pt(0)).square()+(triBar.col(1)-cur_pt(1)).square()+(triBar.col(2)-cur_pt(2)).square();
    it=argSort(tmp);
    while((i0<triMat.rows())&&(notFound)){
      int i = it[i0];
      iv0=triMat(i,0);
      iv1=triMat(i,1);
      iv2=triMat(i,2);
      w=rayTriangleIntersect(sphPts.row(k),nodeMat.row(iv0),nodeMat.row(iv1),nodeMat.row(iv2));
      if(w(0)>0){
        res(k,0)=i; // Index of the triangle 
        res(k,1)=w(0); 
        res(k,2)=w(1); // barycentric coordinate of iv1
        res(k,3)=w(2); // barycentric coordinate of iv2
        notFound=false;
      }
      i0+=1;
    }
    
    // if(notFound){
    //     Rcpp::Rcout<<k<<" not found\n";
    // }
    
  }
  
  return res;
  
}



// [[Rcpp::export]]
Eigen::MatrixXd sphAnglesBarCoord(Eigen::ArrayXXd& sphPtsAngles, Eigen::MatrixXd& nodeMat, Eigen::ArrayXXi& triMat){

  // Angles to Cartesian coordinates
  Eigen::MatrixXd sphPts(sphPtsAngles.rows(), 3);
  sphPts.col(0)=(sphPtsAngles.col(0).array().sin())*(sphPtsAngles.col(1).array().cos());
  sphPts.col(1)=(sphPtsAngles.col(0).array().sin())*(sphPtsAngles.col(1).array().sin());
  sphPts.col(2)=(sphPtsAngles.col(0).array().cos());

  return sphBarCoord( sphPts, nodeMat, triMat);

}





Eigen::ArrayXd projFuncPts(Eigen::MatrixXd& barCoordPts, Eigen::ArrayXd& funcVal, Eigen::ArrayXXi& triMat){

  Eigen::ArrayXd res(barCoordPts.rows());

  int indTri=0;
  for(int i=0; i<barCoordPts.rows();++i){
    indTri=barCoordPts(i,0);
    res(i)=barCoordPts(i,2)*funcVal(triMat(indTri,1));
    res(i)+=barCoordPts(i,3)*funcVal(triMat(indTri,2));
    res(i)+=(1.0-barCoordPts(i,2)-barCoordPts(i,3))*funcVal(triMat(indTri,0));
  }

  return res;

}






