#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <Rcpp.h>
#include <RcppEigen.h>
#include "triSphereBar.h"

// Alias for ease of use
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// Structure for Ball Tree Node
struct BallTreeNode {
  VectorXd center;          // Center of the ball
  double radius;            // Radius of the ball
  std::vector<int> points;  // Indices of points in this ball
  BallTreeNode* left;       // Left child
  BallTreeNode* right;      // Right child
  
  // Constructor
  BallTreeNode() : radius(0), left(nullptr), right(nullptr) {}
};

class BallTree {
public:
  BallTreeNode* root;
  const MatrixXd& data;
  
  // Constructor
  BallTree(const MatrixXd& data) : data(data) {
    std::vector<int> indices(data.rows());
    for (int i = 0; i < data.rows(); ++i) indices[i] = i;
    root = buildTree(indices);
  }
  
  // Destructor to free memory
  ~BallTree() { deleteTree(root); }
  
  // Build the Ball Tree recursively
  BallTreeNode* buildTree(const std::vector<int>& indices) {
    if (indices.empty()) return nullptr;
    
    // Create a new node
    BallTreeNode* node = new BallTreeNode();
    
    // Compute the center of the ball (centroid of points)
    node->center = computeCentroid(indices);
    
    // Compute the radius of the ball
    node->radius = computeRadius(indices, node->center);
    
    // Base case: if there's only one point, it's a leaf node
    if (indices.size() == 1) {
      node->points = indices;
      return node;
    }
    
    // Split the points into two groups
    std::vector<int> leftIndices, rightIndices;
    splitPoints(indices, node->center, leftIndices, rightIndices);
    
    // Recursively build left and right subtrees
    node->left = buildTree(leftIndices);
    node->right = buildTree(rightIndices);
    
    return node;
  }
  
  // Compute the centroid of given points
  VectorXd computeCentroid(const std::vector<int>& indices) {
    VectorXd centroid = VectorXd::Zero(data.cols());
    for (int idx : indices) {
      centroid += data.row(idx);
    }
    centroid /= (indices.size()+0.0);
    return centroid;
  }
  
  // Compute the radius of the ball
  double computeRadius(const std::vector<int>& indices, const VectorXd& center) {
    double maxDist = 0;
    for (int idx : indices) {
      double dist = (data.row(idx) - center.transpose()).norm();
      if (dist > maxDist) maxDist = dist;
    }
   //Rcpp::Rcout<<indices.size()<<" | "<<indices[0]<<" | "<<indices[1]<<" || ";
    return maxDist;
  }
  
  // Split points into two groups based on their distance to the centroid
  void splitPoints(const std::vector<int>& indices, const VectorXd& center,
                   std::vector<int>& left, std::vector<int>& right) {
    VectorXd direction = data.row(indices[0]) - center.transpose();
    for (int idx : indices) {
      if ((data.row(idx) - center.transpose()).dot(direction) > 0) {
        left.push_back(idx);
        //Rcpp::Rcout<<" Left: "<<idx<<" | ";
      } else {
        right.push_back(idx);
        //Rcpp::Rcout<<" Right: "<<idx<<" | ";
      }
    }
    //Rcpp::Rcout<<" || ";
  }
  
  // Recursively delete the tree to free memory
  void deleteTree(BallTreeNode* node) {
    if (!node) return;
    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
  }
  
  // Nearest neighbor search for K closest points
  void kNearestNeighborsSearch(const VectorXd& query, BallTreeNode* node,
                               std::priority_queue<std::pair<double, int>>& pq, int k) {
    if (!node) return;
    
    // Check if this node is a leaf
    if (node->points.size() == 1) {
      int idx = node->points[0];
      double dist = (data.row(idx) - query.transpose()).norm();
      
      if (pq.size() < k) {
        pq.emplace(dist, idx);
      } else if (dist < pq.top().first) {
        pq.pop();
        pq.emplace(dist, idx);
      }
      return;
    }
    
    // Calculate the distance from query point to the center of the ball
    double distToCenter = (node->center - query).norm();
    
    // Prune branches that are too far away
    if (distToCenter - node->radius > (pq.size() < k ? std::numeric_limits<double>::max() : pq.top().first)) return;
    
    // Search left and right subtrees
    kNearestNeighborsSearch(query, node->left, pq, k);
    kNearestNeighborsSearch(query, node->right, pq, k);
  }
  
  // Public interface for k-nearest neighbors search
  std::vector<int> kNearestNeighbors(const VectorXd& query, int k) {
    std::priority_queue<std::pair<double, int>> pq; // Max heap
    
    // Perform search
    kNearestNeighborsSearch(query, root, pq, k);
    
    // Extract results from priority queue
    std::vector<int> neighbors;
    while (!pq.empty()) {
      neighbors.push_back(pq.top().second);
      pq.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end()); // Closest points first
    return neighbors;
  }
};




// [[Rcpp::export]]
Eigen::MatrixXd sphBarCoordBT(Eigen::MatrixXd& sphPts, Eigen::MatrixXd& nodeMat, Eigen::ArrayXXi& triMat,
                                Eigen::MatrixXd& triBar){
  
  Eigen::MatrixXd res=Eigen::MatrixXd::Zero(sphPts.rows(),4);
  Eigen::VectorXd w;
  
  int iv0, iv1, iv2;
  Eigen::VectorXd cur_pt=Eigen::VectorXd::Zero(sphPts.cols());
  Eigen::ArrayXd tmp=Eigen::ArrayXd::Zero(triBar.rows());

  //Rcpp::Rcout<<"Computing Ball Tree\n";
  BallTree tree(triBar);
  //Rcpp::Rcout<<"Finished computing Ball Tree\n";
  
  for(int k=0; k<sphPts.rows(); ++k){
    bool notFound=true;
    int i0=0;
    cur_pt=sphPts.row(k);
    //tmp=(triBar.col(0)-cur_pt(0)).square()+(triBar.col(1)-cur_pt(1)).square()+(triBar.col(2)-cur_pt(2)).square();
    std::vector<int> it=tree.kNearestNeighbors(cur_pt.transpose(), 10);
    
    while((i0<it.size())&&(notFound)){
      int i = it[i0];
      //Rcpp::Rcout<<" "<<i<<" | ";
      iv0=triMat(i,0);
      iv1=triMat(i,1);
      iv2=triMat(i,2);
      w=rayTriangleIntersect(cur_pt,nodeMat.row(iv0),nodeMat.row(iv1),nodeMat.row(iv2));
      if(w(0)>0){
        res(k,0)=i; // Index of the triangle 
        res(k,1)=w(0); 
        res(k,2)=w(1); // barycentric coordinate of iv1
        res(k,3)=w(2); // barycentric coordinate of iv2
        notFound=false;
      }
      i0+=1;
    }
    //Rcpp::Rcout<<" || ";
    
    
    // if(notFound){
    //     Rcpp::Rcout<<k<<" not found\n";
    // }
    
  }
  
  return res;
  
}

// [[Rcpp::export]]
std::vector<int> BT(Eigen::MatrixXd& triBar,Eigen::VectorXd query){

  
  Rcpp::Rcout<<"Computing Ball Tree\n";
  BallTree tree(triBar);
  Rcpp::Rcout<<"Finished computing Ball Tree\n";
  
  // Perform search for 10 nearest neighbors
  std::vector<int> nearestNeighbors = tree.kNearestNeighbors(query, 10);
  
  return nearestNeighbors;
  
}


// [[Rcpp::export]]
int main() {
  // Example 3D data points
  MatrixXd data(8, 3);
  data << 1.0, 2.0, 3.0,
          2.0, 3.0, 4.0,
          3.0, 1.0, 2.0,
          6.0, 5.0, 4.0,
          7.0, 8.0, 9.0,
          5.0, 5.0, 5.0,
          9.0, 7.0, 6.0,
          4.0, 4.0, 4.0;
  
  BallTree tree(data);
  
  // Query point in 3D
  VectorXd query(3);
  query << 3.0, 2.5, 3.5;
  
  // Perform search for 10 nearest neighbors
  std::vector<int> nearestNeighbors = tree.kNearestNeighbors(query, 10);
  
  // Print the indices and coordinates of the nearest neighbors
  std::cout << "10 Closest Neighbors:" << std::endl;
  for (int idx : nearestNeighbors) {
    std::cout << "Index: " << idx << ", Point: " << data.row(idx) << std::endl;
  }
  
  return 0;
}
