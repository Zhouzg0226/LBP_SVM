/************************************************************
ע�⣺
������ʹ�õ�Train.txt��Test.txtÿһ�еĸ�ʽΪ:
ͼ��ľ���·�� ���
�磺
/home/0010.jpg 0
*************************************************************/
#include"LBP.h"

#define CELL_SIZE   128

int main(int argc, char *argv[])
{
	LBP lbp;
	Mat srcImage = imread("aqyh.jpg", 0);
	// extract feature
	double time1_ExtractFeature = getTickCount();
	Mat featureVectorOfTestImage;
	lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVectorOfTestImage);
	if (featureVectorOfTestImage.empty())
		return -1;
	double time2_ExtractFeature = getTickCount();

	return 0;

}
