/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting
#pragma once

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	// Add inliers
  	float scatter = 0.6;
  	for(int i = -5; i < 5; i++)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = i+scatter*rx;
  		point.y = i+scatter*ry;
  		point.z = 0;

  		cloud->points.push_back(point);
  	}
  	// Add outliers
  	int numOutliers = 10;
  	while(numOutliers--)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = 5*rx;
  		point.y = 5*ry;
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}


pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);
  	return viewer;
}

// std::unordered_set<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
// {
// 	int num_pts = cloud->points.size();
// 	std::unordered_set<int> inliersResult;
// 	srand(time(NULL));
  
// 	for(size_t i = 0; i < maxIterations; i++)
// 	{
// 		std::unordered_set<int> inliersTemp;

// 		while(inliersTemp.size() < 2)
// 		{
// 			inliersTemp.insert(rand() % num_pts);
// 		}

// 		pcl::PointXYZ pt1;
// 		pcl::PointXYZ pt2;

// 		auto it = inliersTemp.begin();
// 		pt1 = cloud->points[*it];
// 		it++;
// 		pt2 = cloud->points[*it];
      
//       	int x1 = pt1.x, x2 = pt2.x;
//       	int y1 = pt1.y, y2 = pt2.y;
//       	int z1 = pt1.z, z2 = pt2.z;
      
//         double A = (y1 - y2);
//         double B = (x2 - x1);
//         double C = (x1 * y2 - x2 * y1);
		
// 		for(size_t j = 0; j < num_pts; j++)
// 		{
// 			if(inliersTemp.count(j) > 0) continue;

// 			auto other = cloud->points[j];
//           	auto x3 = other.x;
//           	auto y3 = other.y;

// 			double distance = fabs(A * x3 + B * y3 + C) / sqrt(A*A + B*B);

// 			if(distance <= distanceTol)
// 			{
// 				inliersTemp.insert(j);
// 			}
// 		}

// 		if(inliersTemp.size() > inliersResult.size())
// 		{
// 			inliersResult = inliersTemp;
// 		}
// 	}

// 	return inliersResult;
// }

// std::unordered_set<int> RansacPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
// {
// 	int num_pts = cloud->points.size();
// 	std::unordered_set<int> inliersResult;
// 	srand(time(NULL));
  
// 	for(size_t i = 0; i < maxIterations; i++)
// 	{
// 		std::unordered_set<int> inliersTemp;

// 		while(inliersTemp.size() < 3)
// 		{
// 			inliersTemp.insert(rand() % num_pts);
// 		}

// 		pcl::PointXYZ pt1;
// 		pcl::PointXYZ pt2;
//       	pcl::PointXYZ pt3;

		// 		auto it = inliersTemp.begin();
		// 		pt1 = cloud->points[*it];
		// 		it++;
		// 		pt2 = cloud->points[*it];
//       	it++;
//       	pt3 = cloud->points[*it];
      
//       	int x1 = pt1.x, x2 = pt2.x, x3 = pt3.x;
//       	int y1 = pt1.y, y2 = pt2.y, y3 = pt3.y;
//       	int z1 = pt1.z, z2 = pt2.z, z3 = pt3.z;
      
//       	// A is equivalent to i, the x component of the normal vector of the plane formed by pt1, pt2, and pt3
//       	// B is equivalent to j, the y component
//       	// C is equivalent to k, the z component
//         double A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
//         double B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
//         double C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
//       	double D = -(A * x1 + B * y1 + C * z1);
		
// 		for(size_t j = 0; j < num_pts; j++)
// 		{
// 			if(inliersTemp.count(j) > 0) continue;

// 			auto other = cloud->points[j];
//           	auto x4 = other.x;
//           	auto y4 = other.y;
//           	auto z4 = other.z;
          
// 			double distance = fabs(A * x4 + B * y4 + C * z4 + D) / sqrt(A*A + B*B + C*C);

// 			if(distance <= distanceTol)
// 			{
// 				inliersTemp.insert(j);
// 			}
// 		}

// 		if(inliersTemp.size() > inliersResult.size())
// 		{
// 			inliersResult = inliersTemp;
// 		}
// 	}

// 	return inliersResult;
// }

int main ()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();
	

	// TODO: Change the max iteration and distance tolerance arguments for Ransac function
	std::unordered_set<int> inliers = RansacPlane(cloud, 10, 0.5);

	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for(int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if(inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}


	// Render 2D point cloud with inliers and outliers
	if(inliers.size())
	{
		renderPointCloud(viewer,cloudInliers,"inliers",Color(0,1,0));
  		renderPointCloud(viewer,cloudOutliers,"outliers",Color(1,0,0));
	}
  	else
  	{
  		renderPointCloud(viewer,cloud,"data");
  	}
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}
  	
}
