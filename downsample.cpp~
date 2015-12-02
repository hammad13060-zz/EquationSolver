#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <typeinfo>
#include <cmath>


using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	const char* filename = argc >=2 ? argv[1] : "lena.jpg";
	Mat I = imread(filename, IMREAD_GRAYSCALE);
    if( I.empty())
        return -1;

	Mat dst;
	/*while(true) {
		pyrDown(I, dst, Size(I.cols/2,I.rows/2));
		Size sz = dst.size();
		if (sz.height <= 32 || sz.width <= 32) break;
		I = dst;
	}*/


	resize(I, dst, Size(100,100));
	cout << dst.size().height << " " << dst.size().width << endl;

	namedWindow( "Display window", WINDOW_AUTOSIZE );  
    imshow( "Display window", I);

	waitKey(0);


	return 0;
}
