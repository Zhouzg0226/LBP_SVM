
#ifndef __CLASSIFY__
#define __CLASSIFY__
#include<fstream>
#include "LBP.h"
#include <opencv2/core/types.hpp>
#include<vector>
#include "contrib.hpp"

using namespace std;
using namespace cv::ml;



#define CELL_SIZE   4


class Classify
{
public:
	Classify(string  &filepath,
		SVM::Types svmType, // See SVM::Types. Default value is SVM::C_SVC.
		SVM::KernelTypes kernel,
		double c, // For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0.
		double coef,  // For SVM::POLY or SVM::SIGMOID. Default value is 0.
		double degree, // For SVM::POLY. Default value is 0.
		double gamma, // For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1.
		double nu,  // For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0.
		double p // For SVM::EPS_SVR. Default value is 0.
		);

	~Classify();


public:
	long GetImagesData(string  &filepath, vector<string> *imgPaths, vector<string> *imgClassName, vector<int> *imgLabel);

	long Train();
	long Predict(cv::Mat &img, string *imgClass);

	
private:
	LBP m_lbp;
	cv::Ptr<cv::ml::SVM> m_svm;

	vector<string> m_imgPaths;
	vector<string> m_imgClassName;
	vector<int> m_imgLabel;
	string m_svmModelFilePath;
};



#endif