//指尖检测
#include <iostream>
#include <string>
#include <cmath>
#include <vector>	
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;
//RNG rng(12345);
void skinExtract(const Mat &frame, Mat1b &skinArea);//肤色提取，skinArea为二值化肤色图像
int main(int argc, char** argv)
{
	string nemePicture("data/12.jpg");
	Mat im = imread(nemePicture);

	//肤色提取
	Mat1b skinArea(im.rows, im.cols);

	skinExtract(im, skinArea);

	// Draw contours + hull results
	Mat drawing = Mat::zeros(im.size(), CV_8UC3);
	im.copyTo(drawing);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//寻找轮廓
	findContours(skinArea, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// 找到最大的轮廓（手掌轮廓）
	int index;
	double area, maxArea(0);
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(Mat(contours[i]));
		if (area > maxArea)
		{
			maxArea = area;
			index = i;
		}
	}
	drawContours(drawing, contours, index, Scalar(0, 0, 255), 2, 8, hierarchy);

	vector<int> handHull;// Int type hull
	vector<Vec4i> handDefects;// Convexity defects
	convexHull(Mat(contours[index]), handHull, false);
	convexityDefects(Mat(contours[index]), handHull, handDefects);
	for (vector<Vec4i>::iterator it = handDefects.begin(); it != handDefects.end(); ++it)
	{
		int startidx = (*it)[0];
		Point ptStart(contours[index][startidx]);// point of the contour where the defect begins
		int endidx = (*it)[1];
		Point ptEnd(contours[index][endidx]);// point of the contour where the defect ends
		int faridx = (*it)[2];
		Point ptFar(contours[index][faridx]);// the farthest from the convex hull point within the defect
		int depth = (*it)[3] / 256;// distance between the farthest point and the convex hull

		if (depth > 20)
		{
			line(drawing, ptStart, ptFar, CV_RGB(0, 255, 0), 2);
			line(drawing, ptEnd, ptFar, CV_RGB(0, 255, 0), 2);
			circle(drawing, ptStart, 6, Scalar(255, 0, 0), CV_FILLED);//fingerTip
			circle(drawing, ptEnd, 6, Scalar(255, 0, 0), CV_FILLED);//fingerTip
			circle(drawing, ptFar, 4, Scalar(100, 0, 255), 2);
		}
	}

	namedWindow("helloCV");
	imshow("helloCV", drawing);
	waitKey();
	imwrite("result/result12.png", drawing);
	destroyWindow("helloCV");
	printf("hello\n");
	return 0;
}

void skinExtract(const Mat &im, Mat1b &skinArea)
{
	//转换为YCrCb颜色空间
	Mat YCrCb;
	cvtColor(im, YCrCb, CV_BGR2YCrCb);

	//将多通道图像分离为多个单通道图像
	vector<Mat> planes;
	split(YCrCb, planes);

	Mat m0 = planes[0], m1 = planes[1], m2 = planes[2];

	//运用迭代器访问矩阵元素 
	MatIterator_<uchar> it_Cb = planes[1].begin<uchar>(), it_Cb_end = planes[1].end<uchar>();
	MatIterator_<uchar> it_Cr = planes[2].begin<uchar>();
	MatIterator_<uchar> it_skin = skinArea.begin();

	//人的皮肤颜色在YCbCr色度空间的分布范围:130<=Cb<=170, 85<=Cr<=125
	for (; it_Cb != it_Cb_end; ++it_Cr, ++it_Cb, ++it_skin)
	{
		if (85 <= *it_Cr &&  *it_Cr <= 125 && 130 <= *it_Cb &&  *it_Cb <= 170)
			*it_skin = 255;
		else
		{
			*it_skin = 0;
		}
	}
	medianBlur(skinArea, skinArea, 3);

	//膨胀和腐蚀，膨胀可以填补凹洞（将裂缝桥接），腐蚀可以消除细的凸起（“斑点”噪声）
	dilate(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));
	erode(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));
}