#include <opencv2/core/core.hpp>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <typeinfo>
#include <vector>

using namespace cv;
using namespace std;

typedef struct node {
	int x1, x2, y1, y2;
} ppoint;


vector<ppoint> findClusters(Mat* image);
ppoint expand(Mat* image, Mat* visited, int cc, int* size, int x, int y, int height, int width);
bool validPoint(int x, int y, int X, int Y);
int max(int x, int y);
int min(int x, int y);

#define THRESH_CLUSTER_SIZE 10
int main(int argc, char** argv) {

	namedWindow( "Display window", WINDOW_AUTOSIZE );
	const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    	Mat Image = imread(filename, IMREAD_GRAYSCALE);


	threshold(Image, Image, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
	imshow("Display window", Image);
	waitKey(0);

	vector<ppoint> clusters = findClusters(&Image);

	
	for (int i = 0; i < clusters.size(); i++) {
		ppoint pt = clusters[i];
		Mat roi(Image, Rect(pt.x1-1, pt.y1-1, pt.x2 - pt.x1 + 2, pt.y2 - pt.y1 + 2));  
    		imshow( "Display window", roi);
		waitKey(0);
	}
	destroyWindow("Display window");


	/*for (int i = 50; i < 100; i++) {
		for (int j = 50; j < 80; j++ ) {
			int intensity = (int)Image.at<uchar>(i,j);
			cout << intensity << " ";
		}

		cout << "\n";
	}*/


	/*namedWindow( "Display window", WINDOW_AUTOSIZE );  
    	imshow( "Display window", roi);
	waitKey(0);
	destroyWindow("Display window");*/

	return 0;
}



vector<ppoint> findClusters(Mat* image) {

	vector<ppoint> clusters;

	Mat I = (*image);

	Size size = I.size();
	int height = size.height;
	int width = size.width;
	int cc = 1;

	Mat visited = Mat::zeros(size, CV_8UC1);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			//if unvisited and white expand along that pixel
			if ((*image).at<uchar>(j,i) == 255 && visited.at<uchar>(j,i) == 0) {
				int size = 0;
				ppoint rect = expand(image, &visited, cc, &size, j, i, height, width);

				/*if (size >= THRESH_CLUSTER_SIZE)*/ clusters.push_back(rect);

				cc++;
			}			
		}
	}
	return clusters;	
}


ppoint expand(Mat* image, Mat* visited, int cc, int* size, int y, int x, int height, int width) {
	Mat I = (*image);
	Mat V = (*visited);

	V.at<int>(y,x) = cc;
	*size = (*size) + 1;
	
	ppoint rect;
	rect.x1 = 1000000;
	rect.x2 = -1;
	rect.y1 = 1000000;
	rect.y2 = -1;

	int i = x-1;

	for (;i < x+2; i++) {
		int j = y-1;
		for (; j < y+2; j++) {

			//if pixel cordinate is valid white and unvisited expand along it
			if (validPoint(i,j, width, height) && (*image).at<uchar>(j,i) == 255 && (*visited).at<uchar>(j,i) == 0) {
				ppoint tmp_rect = expand(image, visited, cc, size, j, i, height, width);
				//finding diagonal co-ordinates for generating regions of interest later
				rect.x1 = min(rect.x1, tmp_rect.x1);
				rect.y1 = min(rect.y1, tmp_rect.y1);
				rect.x2 = max(rect.x2, tmp_rect.x2);
				rect.y2 = max(rect.y2, tmp_rect.y2);
			}
		}
	}

	rect.x1 = min(rect.x1, x);
	rect.y1 = min(rect.y1, y);
	rect.x2 = max(rect.x2, x);
	rect.y2 = max(rect.y2, y);

	return rect;
}

bool validPoint(int x, int y, int X, int Y) {
	if ((x >= 0 && x < X) && (y >= 0 && y < Y)) return true;
	return false;
}

int max(int x, int y) {
	if (x >= y) return x;
	return y;
}

int min(int x, int y) {
	if (x <= y) return x;
	return y;
}
