#include <iostream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <algorithm> //min函数
#include <chrono>    //时间
#include <cmath>
#include <iomanip> //输出格式
#include <math.h>
#include <stdio.h>
#include <time.h> //时间
#include <valarray>
using namespace std;

#include <agv_detection_and_tracking/lidar_grad.h> //要用到 msg 中定义的数据类型
#

//ros
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h> //  加载pcd文件
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/extract_clusters.h> //分割中的抽取聚类
#include <pcl/segmentation/sac_segmentation.h> //平面分割
#include <pcl_ros/point_cloud.h>               //可直接将pcl::PointCloud<T>发布出去，不需要转化
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
//#include <pcl / features / normal_3d.h>
//#include <pcl / kdtree / flann.h>
#include <pcl/kdtree/impl/io.hpp>
#include <pcl/kdtree/io.h>
#include <pcl/kdtree/kdtree.h>
//#include <pcl / kdtree / kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/kdtree.h>

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

//Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <queue> //队列依赖库
using namespace Eigen;

//保存车辆的长宽数据,因为检测的不一定那么准，所以长宽给一个范围，以便于后面进行判断
struct car
{
     float width_min;
     float width_max;
     float height_min;
     float height_max;
};
float agv_width = 1.8;
float agv_height = 4.0;
float plank_lelngth = 3.7;
float plank_left_to_gravity_a = 1.43;//质心距板的直线距离
float plank_left_to_gravity_b = 1.24;//质心沿板距板的左边沿点的直线距离

car agvShape = {1.3, 2.2, 3.5, 4.5}; //车辆的长宽数据判断范围

const int dataSize = 1141; //一帧数据里面由多少个数据

struct timeval tv;

pcl::PointCloud<pcl::PointXYZ> background_cloud; //保存背景的点云全局变量
int backgroundNum = 1;                           //取一帧数据当作背景点
float backgroundThes = 0.3;                      //与背景点的距离在这个阈值以内则当成背景点去除
ros::Publisher pub;                              //发布topic声明
ros::Publisher pub_corner;                       //发布拐点和中心点
ros::Subscriber sub;
ros::Publisher pub_cluster_point; //发布聚类的质心
ros::Publisher pub_lidar_grad;    //发布lidar消息

/*********跟踪的代码这里面不要发布，只需要将端点发布即可***********************************/
//生成一个数组，保存五帧符合此车辆特征的中心点坐标，每次得到一个新的车辆中心坐标的时候，需要与此数组的所有点的坐标进行距离的运算，如果距离在半个车宽以内，就表示仍然是这个车辆，然后将数组第一个数据抛弃，尾部加入此数据,并且将这个坐标点以marker的形式发出，并将其以点的形式发出，供订阅。
//当出现一个新的目标车辆的中心点时，不能即刻删除所有的点，加入此点，而应该将此点加入new_car_trajector数组里面，连续判断5帧，如果5帧的距离判断都满足要求，第一帧和第三帧的质心的距离要大于一个阈值（0.05），则表示是一个运动的新的车辆，则将car_trajector里面的点赋为0，将new_car_trajector里面的点放入car_trajector数组
//如果在10帧以内都没有检测到符合要求的聚类物，则将car_traject和new_car_trajector里面的坐标全部赋值为0，表示此时没有车辆

const int car_traject_num = 5;
const int new_car_trajector_num = 5;
pcl::PointXYZ car_traject[car_traject_num];
pcl::PointXYZ new_car_trajector[new_car_trajector_num];

//保存板的两个端点,然后将其发布出来
pcl::PointXYZ car_endpoint[2];

int karman_initial = 0;
double time1 = 0;
double time2 = 0;
//int time_initial=0;
//chrono::steady_clock::time_point t1;

//根据第一帧的数据清除现在的背景点
void background_save(const sensor_msgs::PointCloud2ConstPtr &input);
pcl::PointCloud<pcl::PointXYZ> background_delete(const pcl::PointCloud<pcl::PointXYZ> background_cloud, pcl::PointCloud<pcl::PointXYZ> raw_cloud); //去除背景点
//ros callback function
void callback(const sensor_msgs::PointCloud2ConstPtr &input);

float two_distance(pcl::PointXYZ p1, pcl::PointXYZ p2); //计算两个点的距离
//此函数只提供xy平面的点的映射到直线上
pcl::PointXYZ point_map(pcl::PointXYZ p, float k, float b);

int main(int argc, char **argv)
{
     // Initialize ROS
     ros::init(argc, argv, "agv_detection_tracking");
     ros::NodeHandle nh;
     uint32_t queue_size = 1;

     pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_cloud", queue_size); //发布移动目标聚类点云
     pub_corner = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("pcl_corner", queue_size);
     pub_cluster_point = nh.advertise<visualization_msgs::MarkerArray>("pub_gravity_point", 10);
     pub_lidar_grad = nh.advertise<agv_detection_and_tracking::lidar_grad>("lidar_info", 1); //创建 publisher 对象
     sub = nh.subscribe<sensor_msgs::PointCloud2>("cloud", queue_size, background_save);     //订阅Lidar实时点云
     while (backgroundNum)
     {

          ros::spinOnce();
     }

     sub = nh.subscribe<sensor_msgs::PointCloud2>("cloud", queue_size, callback); //订阅Lidar实时点云

     while (ros::ok())
     {
          ros::spinOnce();
     }

     return 0;
}

double distance(Point2d a, Point2d b)
{
     return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void background_save(const sensor_msgs::PointCloud2ConstPtr &input)
{
     pcl::fromROSMsg(*input, background_cloud); //sensor_msgs::PointCloud2格式转化成pcl::PointCloud<pcl::PointXYZ>
     --backgroundNum;                           //跳出此函数的标志
}

pcl::PointCloud<pcl::PointXYZ> background_delete(const pcl::PointCloud<pcl::PointXYZ> background_cloud, pcl::PointCloud<pcl::PointXYZ> raw_cloud) //去除背景点
{
     pcl::PointCloud<pcl::PointXYZ> filtered_cloud; //保存去除背景点之后的点云
     filtered_cloud.width = dataSize;
     filtered_cloud.height = 1;
     filtered_cloud.points.resize(filtered_cloud.width * filtered_cloud.height);

     //初始化
     for (size_t i = 0; i < dataSize; ++i)
     {
          filtered_cloud.points[i].x = 0;
          filtered_cloud.points[i].y = 0;
          filtered_cloud.points[i].z = 0;
     }

     int dataIter = 0;
     float dist = 0;
     for (size_t i = 0; i < dataSize; ++i)
     {
          pcl::PointXYZ p1 = pcl::PointXYZ(raw_cloud.points[i].x, raw_cloud.points[i].y, raw_cloud.points[i].z);
          pcl::PointXYZ p2 = pcl::PointXYZ(background_cloud.points[i].x, background_cloud.points[i].y, background_cloud.points[i].z);
          dist = two_distance(p1, p2);
          if (dist > backgroundThes) //不是背景点
          {
               filtered_cloud.points[dataIter].x = raw_cloud.points[i].x;
               filtered_cloud.points[dataIter].y = raw_cloud.points[i].y;
               filtered_cloud.points[dataIter].z = raw_cloud.points[i].z;
               ++dataIter;
          }
     }
     return filtered_cloud;
}

float two_distance(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
     float distance = 0;
     distance = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
     return distance;
}
//ros callback function

void callback(const sensor_msgs::PointCloud2ConstPtr &input)
{

     //if (n==0) {time0=input->header.stamp.sec + input->header.stamp.nsec/1000000000.0;n++;}
     //chrono::system_clock::time_point t2   = chrono::system_clock::now();
     time_t tt;
     time(&tt);
     tm *t = localtime(&tt); //这三行用来记录计算机处理雷达数据的时间，在被注释后续输出中有体现，因为当时不需要

     time1 = time2;
     time2 = input->header.stamp.sec + input->header.stamp.nsec / 1000000000.0;
     //cout<<time1<<","<<time2<<endl;
     //std::cout << input->header.seq << std::endl;

     pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
     pcl::fromROSMsg(*input, pcl_cloud); //sensor_msgs::PointCloud2格式转化成pcl::PointCloud<pcl::PointXYZ>
     //1111111111111111111111
     pcl_cloud = background_delete(background_cloud, pcl_cloud); //去除背景点
     //22222222222222222222
     //删除稀疏点：半径滤波器
     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ROI(new pcl::PointCloud<pcl::PointXYZ>);
     for (size_t i = 0; i < dataSize; ++i)
     {
          if ((pcl_cloud.points[i].x > 1e-5) | (pcl_cloud.points[i].y > 1e-5) | (pcl_cloud.points[i].y > 1e-5)) //若不全为0
          {
               pcl::PointXYZ p;
               p.x = pcl_cloud.points[i].x;
               p.y = pcl_cloud.points[i].y;
               p.z = pcl_cloud.points[i].z;

               cloud_ROI->points.push_back(p);
          }
     }
     /* 
     sensor_msgs::PointCloud2 roimsg;
     pcl::toROSMsg(*cloud_ROI, roimsg);
     roimsg.header.frame_id = "velodyne1";
     pub.publish(roimsg);
     */

     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_Radius(new pcl::PointCloud<pcl::PointXYZ>);
     pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier; //创建滤波器
     radiusoutlier.setInputCloud(cloud_ROI);                 //设置输入点云
     radiusoutlier.setRadiusSearch(0.6);                     //设置半径为1的范围内找临近点
     radiusoutlier.setMinNeighborsInRadius(6);               //设置查询点的邻域点集数小于2的删除
     radiusoutlier.filter(*cloud_after_Radius);

     //判断此时去除背景点，滤波之后的点云数据是否是自己想要的
     /* sensor_msgs::PointCloud2 roimsg;
     pcl::toROSMsg(*cloud_after_Radius, roimsg);
     roimsg.header.frame_id = "velodyne1";
     pub.publish(roimsg)*/
     ;

     //发布空的marker删除上一次的质心的marker点
     visualization_msgs::MarkerArray markerArrayDeleteAll;
     for (int j = 0; j < 16; ++j)
     {
          visualization_msgs::Marker marker;
          marker.header.frame_id = "velodyne1";
          marker.header.stamp = ros::Time::now();
          marker.ns = "basic_shapes1";
          marker.id = j;
          marker.action = visualization_msgs::Marker::DELETE;

          markerArrayDeleteAll.markers.push_back(marker);
     }
     pub_cluster_point.publish(markerArrayDeleteAll);

     //3333333333333333333333333333
     //聚类分割
     //为提取点云时使用的搜素对象利用输入点云cloud_after_Radius创建Kd树对象tree
     int minSizeOfCluster = 12;
     if (cloud_after_Radius->points.size() >= minSizeOfCluster) //当滤波之后的剩余点数大于等于一个聚类的最少点数时进行聚类
     {
          vector<pcl::PointXYZ> cluster_gravity_point; //保存每个聚类的质心点,以便于发布marker观察质心的位置

          pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
          tree->setInputCloud(cloud_after_Radius); //创建点云索引向量，用于存储实际的点云信息
          vector<pcl::PointIndices> cluster_indices;
          pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

          ec.setClusterTolerance(0.15);           //设置近邻搜索的搜索半径为0.5m
          ec.setMinClusterSize(minSizeOfCluster); //设置一个聚类需要的最少点数目为minSizeOfCluster
          ec.setMaxClusterSize(25000);            //设置一个聚类需要的最大点数目为25000
          ec.setSearchMethod(tree);               //设置点云的搜索机制
          ec.setInputCloud(cloud_after_Radius);
          ec.extract(cluster_indices); //从点云中提取聚类，并将点云索引保存在cluster_indices中

          //将cluster_indices中的点提取出来

          vector<vector<pcl::PointXYZ>> pointcloud_after_cluster; //保存聚类的点

          for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
          {
               vector<pcl::PointXYZ> single_cluster;
               pcl::PointXYZ single_cluster_point;
               single_cluster_point.x = 0;
               single_cluster_point.y = 0;
               single_cluster_point.z = 0;

               int single_cluster_point_num = 0;
               //创建新的点云数据集cloud_cluster，将所有当前聚类写入到点云数据集中
               for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
               {
                    single_cluster.push_back(cloud_after_Radius->points[*pit]);
                    ++single_cluster_point_num;
                    single_cluster_point.x += cloud_after_Radius->points[*pit].x;
                    single_cluster_point.y += cloud_after_Radius->points[*pit].y;
                    single_cluster_point.z += cloud_after_Radius->points[*pit].z;
               }
               single_cluster_point.x /= single_cluster_point_num;
               single_cluster_point.y /= single_cluster_point_num;
               single_cluster_point.z /= single_cluster_point_num;
               cluster_gravity_point.push_back(single_cluster_point); //保存每个聚类的质心点
               pointcloud_after_cluster.push_back(single_cluster);    //保存聚类后的所有点
          }

          //发布每个聚类点的质心
          visualization_msgs::MarkerArray markerArray;

          for (int j = 0; j < cluster_gravity_point.size(); j++)
          {

               visualization_msgs::Marker marker;
               marker.header.frame_id = "velodyne1";
               marker.header.stamp = ros::Time::now();
               marker.ns = "basic_shapes1";
               marker.action = visualization_msgs::Marker::ADD;
               marker.pose.orientation.w = 1.0;
               marker.id = j;

               marker.type = visualization_msgs::Marker::CUBE;
               marker.color.b = 200;
               marker.color.g = 100;
               marker.color.r = 100;
               marker.color.a = 0.5;
               marker.scale.x = 2;
               marker.scale.y = 2;
               marker.scale.z = 2;

               geometry_msgs::Pose pose;
               pose.position.x = cluster_gravity_point[j].x;
               pose.position.y = cluster_gravity_point[j].y;
               pose.position.z = cluster_gravity_point[j].z;

               marker.pose = pose;
               markerArray.markers.push_back(marker);
          }
          pub_cluster_point.publish(markerArray);

          static int count = 0; //计算聚类多少次
          ++count;
          //cout << "count=" << count << " ";
          if (pointcloud_after_cluster.size() > 0)
               cout << "pointcloud_after_cluster size=" << pointcloud_after_cluster.size() << endl;

          //只需要进行直线拟合，利用I型检测，检测安装的一块平板，假设打在车上的就那块平板
          //对每个类进行RANSAC拟合直线段，因为板比较平，则限制直线拟合的偏差小一点，比如0.1米，但是也必须控制内点的数量在一定范围内，否则不是车辆的
          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_endpoint(new pcl::PointCloud<pcl::PointXYZ>); //保存端点点云

          for (int j = 0; j < pointcloud_after_cluster.size(); ++j)
          {
               //将vector点云转化为pcl点云，进行ransac拟合直线
               pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_clustered_vec(new pcl::PointCloud<pcl::PointXYZ>);

               pcl_clustered_vec->width = pointcloud_after_cluster[j].size();
               pcl_clustered_vec->height = 1;
               pcl_clustered_vec->points.resize(pcl_clustered_vec->width * pcl_clustered_vec->height);

               for (size_t i = 0; i < pointcloud_after_cluster[j].size(); ++i)
               {
                    pcl_clustered_vec->points[i].x = pointcloud_after_cluster[j][i].x;
                    pcl_clustered_vec->points[i].y = pointcloud_after_cluster[j][i].y;
                    pcl_clustered_vec->points[i].z = pointcloud_after_cluster[j][i].z;
               }

               //ransac拟合直线
               pcl::ModelCoefficients::Ptr pcl_clustered_vec_coefficients(new pcl::ModelCoefficients);
               pcl::PointIndices::Ptr pcl_clustered_vec_inliers(new pcl::PointIndices);
               pcl::SACSegmentation<pcl::PointXYZ> pcl_clustered_vec_seg;
               pcl_clustered_vec_seg.setOptimizeCoefficients(true);

               pcl_clustered_vec_seg.setModelType(pcl::SACMODEL_LINE);
               pcl_clustered_vec_seg.setMethodType(pcl::SAC_RANSAC);
               pcl_clustered_vec_seg.setDistanceThreshold(0.1);
               pcl_clustered_vec_seg.setInputCloud(pcl_clustered_vec);
               pcl_clustered_vec_seg.segment(*pcl_clustered_vec_inliers, *pcl_clustered_vec_coefficients);

               //取出拟合内点pcl_clustered_vec_inliers里面的端点
               if (pcl_clustered_vec_inliers->indices.size() > minSizeOfCluster) //里面的点数必须大于minSizeOfCluster才进行寻找直线的端点
               {

                    float k = pcl_clustered_vec_coefficients->values[4] / pcl_clustered_vec_coefficients->values[3];
                    float b = pcl_clustered_vec_coefficients->values[1] - pcl_clustered_vec_coefficients->values[0] * pcl_clustered_vec_coefficients->values[4] / pcl_clustered_vec_coefficients->values[3];

                    //将所有点映射到直线上

                    for (int i = 0; i < pcl_clustered_vec_inliers->indices.size(); ++i)
                    {
                         pcl::PointXYZ p_tmp;
                         p_tmp.x = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].x;
                         p_tmp.y = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].y;
                         p_tmp.z = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].z;
                         p_tmp = point_map(p_tmp, k, b);
                         pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].x = p_tmp.x;
                         pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].y = p_tmp.y;
                         pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].z = p_tmp.z;
                    }

                    //保存端点，以内点的y值作为判断条件，保存y最大最小值的两个点,保存质心点
                    pcl::PointXYZ end_p_min;
                    pcl::PointXYZ end_p_max;
                    pcl::PointXYZ gravity_point;
                    end_p_min.x = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[0]].x;
                    end_p_min.y = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[0]].y;
                    end_p_min.z = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[0]].z;

                    if (pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[1]].y > end_p_min.y)
                    {
                         end_p_max.x = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[1]].x;
                         end_p_max.y = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[1]].y;
                         end_p_max.z = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[1]].z;
                    }
                    else
                    {
                         end_p_max.x = end_p_min.x;
                         end_p_max.y = end_p_min.y;
                         end_p_max.z = end_p_min.z;

                         end_p_min.x = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[1]].x;
                         end_p_min.y = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[1]].y;
                         end_p_min.z = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[1]].z;
                    }
                    for (int i = 2; i < pcl_clustered_vec_inliers->indices.size(); ++i)
                    {
                         if (pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].y > end_p_max.y)
                         {
                              end_p_max.x = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].x;
                              end_p_max.y = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].y;
                              end_p_max.z = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].z;
                              continue;
                         }
                         if (pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].y < end_p_min.y)
                         {
                              end_p_min.x = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].x;
                              end_p_min.y = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].y;
                              end_p_min.z = pcl_clustered_vec->points[pcl_clustered_vec_inliers->indices[i]].z;
                              continue;
                         }
                    }

                    //判断两个端点的距离是不是在车宽的一定范围内，若在，则表示是车辆
                    float endpoint_dist = two_distance(end_p_min, end_p_max);
                    if (endpoint_dist < plank_lelngth + 0.2 && endpoint_dist > plank_lelngth - 0.2)
                    {
                         cloud_endpoint->points.push_back(end_p_min);
                         cloud_endpoint->points.push_back(end_p_max);

                         cout << "end_p_min x=" << end_p_min.x << " y=" << end_p_min.y << " z=" << end_p_min.z << endl;
                         //计算车辆的质心坐标
                         float x1 = end_p_max.x;
                         float y1 = end_p_max.y;
                         float x2 = end_p_min.x;
                         float y2 = end_p_min.y;
                         float a = plank_left_to_gravity_a;
                         float b = plank_left_to_gravity_b;
                         gravity_point.x = a / sqrt(pow((y1 - y2) / (x1 - x2), 2.0) + 1) * ((y2 - y1) / (x1 - x2)) + b / sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0)) * (x1 - x2) + x1;
                         gravity_point.y = a / sqrt(pow((y1 - y2) / (x1 - x2), 2.0) + 1) + b / sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0)) * (y1 - y2) + y1;
                         gravity_point.z = 0;
                         cloud_endpoint->points.push_back(gravity_point);

                         float l = sqrt(pow(gravity_point.x - x1, 2.0) + pow(gravity_point.y - y1, 2.0));
                         //cout << "l=" << l;
                         //cout << " l_yuan=" << sqrt(a * a + b * b) << endl;
                         if(abs(l-sqrt(a * a + b * b))<1e-5)//验证计算的坐标是否正确
                              cout << "******************************************" << endl;
                    }
               }
          }
          if (cloud_endpoint->points.size() > 1)
          {
               //cout << "publish" << endl;
               sensor_msgs::PointCloud2 roimsg;
               pcl::toROSMsg(*cloud_endpoint, roimsg);
               roimsg.header.frame_id = "velodyne1";
               pub.publish(roimsg);

               //创建 lidar 消息
               agv_detection_and_tracking::lidar_grad msg;

               msg.x = (cloud_endpoint->points[0].x + cloud_endpoint->points[1].x) / 2; //只取出cloud_endpoint的一二个点的中点发布出去
               msg.y = (cloud_endpoint->points[0].y + cloud_endpoint->points[1].y) / 2;
               msg.theta = 0;
               msg.vel = 0;
               gettimeofday(&tv, NULL);
               msg.time_sec = tv.tv_sec;
               msg.time_usec = tv.tv_usec;
               //ROS_INFO("time stamp %ld.%ld", msg.time_sec, msg.time_usec); //打印函数，类似 printf()
               pub_lidar_grad.publish(msg); //发布消息
          }
     }
}

//此函数只提供xy平面的点的映射到直线上
pcl::PointXYZ point_map(pcl::PointXYZ p, float k, float b)
{
     pcl::PointXYZ tmp;
     tmp.x = (p.x + k * p.y - k * b) / (k * k + 1);
     tmp.y = (k * p.x + k * k * p.y + b) / (k * k + 1);
     tmp.z = p.z;
     return tmp;
}
