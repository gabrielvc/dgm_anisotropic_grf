


Eigen::MatrixXd sphBarCoord(Eigen::MatrixXd& sphPts, Eigen::MatrixXd& nodeMat, Eigen::ArrayXXi& triMat);

Eigen::MatrixXd sphAnglesBarCoord(Eigen::ArrayXXd& sphPtsAngles, Eigen::MatrixXd& nodeMat, Eigen::ArrayXXi& triMat);

Eigen::ArrayXd projFuncPts(Eigen::MatrixXd& barCoordPts, Eigen::ArrayXd& funcVal, Eigen::ArrayXXi& triMat);

Eigen::VectorXd rayTriangleIntersect(Eigen::VectorXd , Eigen::VectorXd , Eigen::VectorXd , Eigen::VectorXd );
