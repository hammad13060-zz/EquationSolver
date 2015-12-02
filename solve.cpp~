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


#define HORIZONTAL 1
#define VERTICAL 2

typedef struct node {
	int a; 
	int b;
} point;

void mserExtractor (const Mat& image, Mat& mserOutMask);
vector<int> seperationHistogram(Mat *image, int dir);
vector<Mat> seperateByVerticalLine(Mat *image,vector<int> hist);
vector<Mat> seperateByHorizontalLine(Mat *image,vector<int> hist);
void showCharImages(vector<Mat> char_images);
void resizeVector(vector<Mat>& imgs);
Mat characterLBP(Mat I);
unsigned char lbpMask(Mat image, int i, int j);
vector<Mat> lbpVector(vector<Mat> params);

uchar values[] = {128, 64, 32, 1, 0, 16, 2, 4, 8};
Mat mask( 3, 3, CV_8UC1, values);
void myImageShow(Mat image);

int main(int argc, char** argv) {


	String window_name = "binarized_image_window";
	const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    Mat Image = imread(filename, IMREAD_GRAYSCALE);

    //medianBlur(Image, Image, 3);

	Mat bImage = Mat::zeros(Image.size(), CV_8UC1);

	threshold(Image, bImage, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);

	namedWindow( window_name, WINDOW_AUTOSIZE );
	imshow(window_name, bImage);	
	waitKey(0);
	destroyWindow(window_name);

	vector<int> hHistogram = seperationHistogram(&bImage, HORIZONTAL);
	vector<Mat> equation_image_vector = seperateByHorizontalLine(&bImage, hHistogram);

	namedWindow( window_name, WINDOW_AUTOSIZE );
	imshow(window_name, equation_image_vector[0]);
	waitKey(0);
	destroyWindow(window_name);

	namedWindow( window_name, WINDOW_AUTOSIZE );	
	imshow(window_name, equation_image_vector[1]);
	waitKey(0);
	destroyWindow(window_name);

	vector<int> vHistogram_1 = seperationHistogram(&(equation_image_vector[0]), VERTICAL);
	vector<int> vHistogram_2 = seperationHistogram(&(equation_image_vector[1]), VERTICAL);

    	/*int erosion_size = 1;  
    	Mat element = getStructuringElement(MORPH_CROSS,
              Size(2 * erosion_size + 1, 2 * erosion_size + 1),
              Point(erosion_size, erosion_size) );

	int prev = -1;
	int chunk = 0;
	for (int i = 0; i < vHistogram_1.size(); i++) {
		if (vHistogram_1[i] == 0) {
			if (prev == -1) {
				chunk++;
				prev = 0;
			}
		} else prev = -1;
		//cout << i+1 << " <-----> " << vHistogram_1[i] << endl;
	}

	cout << "chunk: " << chunk	 << endl;*/
	

	vector<Mat> params_image_1 = seperateByVerticalLine(&(equation_image_vector[0]), vHistogram_1);
	vector<Mat> params_image_2 = seperateByVerticalLine(&(equation_image_vector[1]), vHistogram_2);

	resizeVector(params_image_1);
	resizeVector(params_image_2);

    
    //showCharImages(params_image_1);
    //showCharImages(params_image_2);
    	

	

    vector<Mat> lbp_params_1 = lbpVector(params_image_1);
    vector<Mat> lbp_params_2 = lbpVector(params_image_2);

    showCharImages(lbp_params_1);
    showCharImages(lbp_params_2);

    
	return 0;
}


vector<int> seperationHistogram(Mat *image, int dir)
{
	int height = (*image).size().height;
	int width = (*image).size().width;

	int len = (dir == HORIZONTAL) ? height : width;
	vector<int> hist(len);

	//initialization
	for (int i = 0; i < len; i++)
		hist[i] = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int intensity = (*image).at<uchar>(i,j);
			if (dir == HORIZONTAL) {
				if (intensity == 255) hist[i] += 1;
			} else if (dir == VERTICAL) {
				if (intensity == 255) hist[j] += 1;
			}
		}
	}
	return hist;	
}


vector<Mat> seperateByHorizontalLine(Mat *image,vector<int> hist) {

	vector<Mat> equation_images(2);

	int level = 0;
	bool state = true; //looking for white after black
	//false is the oposite
	point equation1_y;
	point equation2_y;
	//cout << 0 << endl;
	for (int i = 0; i < hist.size()-1; i++) {
		
		if (state) {
			if (hist[i] == 0 && hist[i+1] > 0) {
				if (level == 0) {
					equation1_y.a = i;
				} else if (level == 1) {
					equation2_y.a = i;
				}
				state = false;
			}
		} else if (!state) {
			if (hist[i] > 0 && hist[i+1] == 0) {
				if (level == 0) {
					equation1_y.b = i;
					level++;
				} else if (level == 1) {
					equation2_y.b = i;
					level++;
				}
				state = true;
			}
		}
	}

	//cout << 0 << endl;

	/*int tmpHeight = equation1_y.b - equation1_y.a + 1;
	int width = (*image).size().width;
	Mat equation1_image = Mat::zeros(tmpHeight, width, CV_8UC1);

	for (int i = 0; i < tmpHeight; i++) {
		for (int j = 0; j < width; j++) {
			int offset = equation1_y.a;
			equation1_image.at<int>(i,j) = (*image).at<int>(i+offset,j);
		}
	}

	cout << 0 << endl;


	tmpHeight = equation2_y.b - equation2_y.a + 1;
	Mat equation2_image = Mat::zeros(tmpHeight, width, CV_8UC1);

	for (int i = 0; i < tmpHeight; i++) {
		for (int j = 0; j < width; j++) {
			int offset = equation2_y.a;
			equation2_image.at<int>(i,j) = (*image).at<int>(i+offset,j);
		}
	}*/
	

	/*equation_images[0] = equation1_image;
	equation_images[1] = equation2_image;*/
	int width = (*image).size().width;
	cout << equation1_y.a << " " << equation1_y.b << endl;
	cout << equation2_y.a << " " << equation2_y.b << endl;
	(new Mat(*image, Rect(0, equation1_y.a, width-1, equation1_y.b-equation1_y.a+1)))->copyTo(equation_images[0]);
	(new Mat(*image, Rect(0, equation2_y.a, width-1, equation2_y.b-equation2_y.a+1)))->copyTo(equation_images[1]);
	

	cout << 0 << endl;

	return equation_images;
}


vector<Mat> seperateByVerticalLine(Mat *image,vector<int> hist){
	
	vector<Mat> char_images;
	bool state = true;
	
	int x1 = 0;
	int x2 = 0;

	int height = (*image).size().height;

	for (int i = 0; i < hist.size()-1; i++) {
		
		if (state) {
			if (hist[i] == 0 && hist[i+1] > 0) {
                cout << hist[i] << " " << hist[i+1] << endl;
				x1 = i;
				state = false;
			}
		} else if (!state) {
			if (hist[i] > 0 && hist[i+1] == 0) {
				x2 = i;
                cout << hist[i] << " " << hist[i+1] << endl;
				Mat char_image(*image, Rect(x1,0,x2-x1+1, height));
				char_images.push_back(char_image);
				state = true;
			}
		}
	}

	return char_images;
}


void showCharImages(vector<Mat> char_images) {
    String window_name = "char_images_window";
	for (int i = 0; i < char_images.size(); i++) {
        namedWindow( window_name, WINDOW_AUTOSIZE );
        if (char_images.empty()) break;	
		imshow(window_name, char_images[i]);
		waitKey(0);
		destroyWindow(window_name);
	}
}

//resize a vector of images to 32 cross 32
void resizeVector(vector<Mat>& images) {

	for (int i = 0; i < images.size(); i++) {
		Mat image = images[i];
		if (image.empty()) break;
		Mat dst;
		resize(image, dst, Size(32, 32));
		cout << dst.size().height << " " << dst.size().width << endl;
		images[i] = dst;		
	}
}


//lbp of 32*32 image divided into regions of 8*8 cells
Mat characterLBP(Mat I) {

	Size size = I.size();

	Mat lbpVector = Mat::zeros(1, 64, CV_8UC4);

	int cell = 0;

    for (int i = 0; i < 32; i += 4) {
        for (int j = 0; j < 32; j += 4) {
            int channel = 0;
            for (int k = 1; k < 3; k++) {
                for (int l = 1; l < 3; l++){
                    unsigned char decimalValue = lbpMask(I, i + k, j + l);
                    lbpVector.at<Vec4b>(0,cell)[channel] = decimalValue;
                    cout << "channel: " << channel << endl;
				    channel++;    
                }
            }
            cell++;
       }
    }

	return lbpVector;
}

unsigned char lbpMask(Mat image, int i, int j) {
	

	unsigned char center = image.at<uchar>(i, j);
	unsigned char result = 0;

	int p = i-1;
	for (; p < i+2; p++){
		int q = j-1;
		for (; q < j+2; q++)
		{
			int neighbour_value = image.at<uchar>(p,q);
			if (center <= neighbour_value)
				result += mask.at<uchar>(p-i+1, q-j+1);			
		}			
	}

	return result;
}

void myImageShow(Mat image) {
    String window_name = "my_image_window";
    namedWindow( window_name, WINDOW_AUTOSIZE );
    imshow(window_name, image);
    cout << image.size().height << " " << image.size().width << endl;
    waitKey(0);
    destroyWindow(window_name);
}


vector<Mat> lbpVector(vector<Mat> params) {
    vector<Mat> lbpParams;
    for (int i = 0; i < params.size(); i++) {
        Mat lbp = params[i];
        if (lbp.empty()) break;
        lbpParams.push_back(characterLBP(lbp));    
    }

    return lbpParams;
}
