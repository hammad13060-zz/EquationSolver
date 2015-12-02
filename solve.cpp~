/*
	IMAGE ANALYSIS PROJECT	
	ASHISH AAPAN (2013024)
	HAMMAD AKHTAR (2013060)
*/


#include <cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <typeinfo>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

//direction for horizontal and vertical histograms
#define HORIZONTAL 1
#define VERTICAL 2

uchar values[3][3] = {{128, 64, 32}, {1, 0, 16}, {2, 4, 8}};
	Mat mask( 3, 3, CV_8UC1, values);

typedef struct node {
	int a; 
	int b;
} point;

//function blue prints
void mserExtractor (const Mat& image, Mat& mserOutMask);
vector<int> seperationHistogram(Mat *image, int dir);
vector<Mat> seperateByVerticalLine(Mat *image,vector<int> hist);
vector<Mat> seperateByHorizontalLine(Mat *image,vector<int> hist);
void showCharImages(vector<Mat> char_images);
void resizeVector(vector<Mat>& imgs);
Mat characterLBP(Mat I);
unsigned char lbpMask(Mat image, int i, int j);
vector<Mat> lbpVector(vector<Mat> params);
vector<uchar> classifyEquation(Ptr<SVM> svm, vector<Mat> params);
void printEquation(vector<uchar> equation, String num);
void myImageShow(Mat image);

int main(int argc, char** argv) {

	//loading the svm classifier
	Ptr<SVM> svm = StatModel::load<SVM>("./res/classifier");

	//defining a window for displaying images
	String window_name = "binarized_image_window";
	const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    Mat Image = imread(filename, IMREAD_GRAYSCALE);

    //medianBlur(Image, Image, 3);

	//container for binary image	
	Mat bImage = Mat::zeros(Image.size(), CV_8UC1);
	
	//applying otsu's thresholding (inverse binary) 
	threshold(Image, bImage, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
	
	//displaying binarized image
	myImageShow(bImage);
	
	//-----------------------seperating the two equations w.r.t horizontal lines-------------------------------------------//
	vector<int> hHistogram = seperationHistogram(&bImage, HORIZONTAL); //histogram for white pixel intensity along each row of image. 
																	   // zero frequecy represents a line
	vector<Mat> equation_image_vector = seperateByHorizontalLine(&bImage, hHistogram); //seperation of equations on the basis of horizontal histogram

	//showing segmented images
	myImageShow(equation_image_vector[0]); //equation 1
	myImageShow(equation_image_vector[1]); //equation 2

	//-------------------------seperating parameters and co-efficients of equations--------------------------------------//

	// histograms for equation 1 and equation 2
	//histogram for white pixel intensity along each row of image. 
	// zero frequecy represents a line
	vector<int> vHistogram_1 = seperationHistogram(&(equation_image_vector[0]), VERTICAL);
	vector<int> vHistogram_2 = seperationHistogram(&(equation_image_vector[1]), VERTICAL);
	

	//seperating characters using vertical histogram
	vector<Mat> params_image_1 = seperateByVerticalLine(&(equation_image_vector[0]), vHistogram_1); //vector of segmented chars of equation 1
	vector<Mat> params_image_2 = seperateByVerticalLine(&(equation_image_vector[1]), vHistogram_2); //vector of segmented chars of equation 2

	//--------------------------samling images to 32*32-----------------------------------------//
	resizeVector(params_image_1); //this function resizes each image in segmented char vectors to 32*32 (equation 1)
	resizeVector(params_image_2); //this function resizes each image in segmented char vectors to 32*32 (equation 2)

    
    //showCharImages(params_image_1);
    //showCharImages(params_image_2);
    	

	

    vector<Mat> lbp_params_1 = lbpVector(params_image_1); //forming Local binary patterns for each image in vector. A vector returned (equation 1)
    vector<Mat> lbp_params_2 = lbpVector(params_image_2); //forming Local binary patterns for each image in vector. A vector returned (equation 2)

    //showCharImages(lbp_params_1);
    //showCharImages(lbp_params_2);

	//-------------------------------------------classification process---------------------------------------------//

	vector<uchar>equation1 = classifyEquation(svm, lbp_params_1); //classification from svm model for equation 1
	vector<uchar>equation2 = classifyEquation(svm, lbp_params_2); //classification from svm model for equation 2

	//--------------------------------------------printing classified equations---------------------------------------//
	printEquation(equation1, "1");
	printEquation(equation2, "2");
	

    
	return 0;
}

//histogram for vertical and horizontal lines
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


//segmenting equations w.r.t to horizontal lines
//for now supports two region segmentation
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

	int width = (*image).size().width;
	cout << equation1_y.a << " " << equation1_y.b << endl;
	cout << equation2_y.a << " " << equation2_y.b << endl;
	(new Mat(*image, Rect(0, equation1_y.a, width-1, equation1_y.b-equation1_y.a+1)))->copyTo(equation_images[0]);
	(new Mat(*image, Rect(0, equation2_y.a, width-1, equation2_y.b-equation2_y.a+1)))->copyTo(equation_images[1]);
	

	cout << 0 << endl;

	return equation_images;
}

//segment symbols w.r.t to vertical lines
//supports segmentation into n regions/symbols
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

//helepr for displaying images in vector
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

//resize each image to 32*32 inside a vector
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
Mat characterLBP(Mat Image) {
	Mat I;
	copyMakeBorder(Image, I, 1,1,1,1, BORDER_CONSTANT, Scalar(0));
	Size size = I.size();
	cout << size.height << " * " << size.width << endl;

	Mat lbpVector = Mat::zeros(1, 64*256, CV_8UC1);
	
	int cell = 0;

    for (int i = 1; i <= 32; i += 4) {
        for (int j = 1; j <= 32; j += 4) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 4; l++){
                    unsigned char decimalValue = lbpMask(I, i + k, j + l);
                    lbpVector.at<uchar>(0,cell*256+decimalValue) = lbpVector.at<uchar>(0,cell*256+decimalValue) + 1;  
                }
            }
            cell++;
       }
    }

	return lbpVector;
}


//local binary pattern mask to be applied on a pizel (row,col)
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

//helper for displaying processed images on the go
void myImageShow(Mat image) {
    String window_name = "my_image_window";
    namedWindow( window_name, WINDOW_AUTOSIZE );
    imshow(window_name, image);
    cout << image.size().height << " " << image.size().width << endl;
    waitKey(0);
    destroyWindow(window_name);
}

//mask for lbp
vector<Mat> lbpVector(vector<Mat> params) {
	
    vector<Mat> lbpParams;
    for (int i = 0; i < params.size(); i++) {
        Mat lbp = params[i];
        if (lbp.empty()) break;
        lbpParams.push_back(characterLBP(lbp));    
    }

    return lbpParams;
}

//classifies images in vector using given svm model
vector<uchar> classifyEquation(Ptr<SVM> svm, vector<Mat> params){
	vector<uchar> equation;

	for (int i = 0; i < params.size(); i++) {
		Mat image = params[i];
		if (image.empty()) break;
		Mat testVector;
		image.convertTo(testVector, CV_32FC1);
		uchar symbol = (uchar)svm->predict(testVector);
		equation.push_back(symbol);
	}

	return equation;
}

//print equations given in a vector. each element representing a symbol.
void printEquation(vector<uchar> equation, String num) {
	cout << "equation" + num + ": ";
	for (int i = 0; i < equation.size(); i++)
		cout << equation[i] << " ";

	cout << "\n";

}
