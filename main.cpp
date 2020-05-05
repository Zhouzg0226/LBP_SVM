
#include"Classify.h"


int main(int argc, char *argv[])
{
        string datapath = "C:\\Users\\zhouz\\Desktop\\MNIST";
	Classify m_classify(datapath,
		SVM::C_SVC,  // svmType
		SVM::LINEAR, // kernel
		1,   // c
		0,   // coef
		0,   // degree
		1,   // gamma
		0,   // nu
		0);  // p

	long lret = m_classify.Train();

	cv::Mat img = imread("3_224.jpg", 0);
	string predictClass;
	lret = m_classify.Predict(img, &predictClass);

	return 0;

}
