#include "Classify.h"



Classify::Classify(string  &filepath,
	SVM::Types svmType, // See SVM::Types. Default value is SVM::C_SVC.
	SVM::KernelTypes kernel,
	double c, // For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0.
	double coef,  // For SVM::POLY or SVM::SIGMOID. Default value is 0.
	double degree, // For SVM::POLY. Default value is 0.
	double gamma, // For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1.
	double nu,  // For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0.
	double p // For SVM::EPS_SVR. Default value is 0.
	)
{

	GetImagesData(filepath, &m_imgPaths, &m_imgClassName, &m_imgLabel);

	m_svm = SVM::create();
	m_svm->setC(c);
	m_svm->setCoef0(coef);
	m_svm->setDegree(degree);
	m_svm->setGamma(gamma);
	m_svm->setKernel(kernel);
	m_svm->setNu(nu);
	m_svm->setP(p);
	m_svm->setType(svmType);
	//svm->setTermCriteria(TermCriteria(TermCriteria::EPS, 1000, FLT_EPSILON)); // based on accuracy
	m_svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6)); // based on the maximum number of iterations

}

Classify::~Classify()
{

}

long Classify::GetImagesData(string  &filepath, vector<string> *imgPaths, vector<string> *imgClassName, vector<int> *imgLabel)
{
	//m_dataFilesList.clear();

	Directory dir;
	std::string exten = "*";
	std::vector<std::string> foldernames = dir.GetListFolders(filepath, exten, true);
	std::vector<std::string> ClassNames = dir.GetListFolders(filepath, exten, false);
	*imgClassName = ClassNames;

	if (foldernames.size() == 0)
	{
		return -1;
	}
	else if (foldernames.size() < 2)
	{
		return -1;
	}

	//m_sameLabelNum.resize(foldernames.size(), 0);

	imgPaths->clear();
	for (int i = 0; i < (int)foldernames.size(); i++)
	{
		vector<cv::String> fn;
		glob(foldernames[i], fn, true);

		int imgNum = fn.size();
		if (imgNum < 1)
		{
			return -1;
		}

		for (auto file : fn)
		{
			imgPaths->push_back(file);
			imgLabel->push_back(i);
		}

	}
	return 0;
}


long Classify::Train()
{
	Mat feaData;
	Mat label;
	int i = 0;
	for (auto imgpath : m_imgPaths)
	{
		cv::Mat srcImage = cv::imread(imgpath.c_str(), 0);
		if (srcImage.empty() || srcImage.depth() != CV_8U)
		{
			return -1;  
		}
		Mat featureVector;
		m_lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVector);
		if (featureVector.empty()) return -2;

		feaData.push_back(featureVector);
		label.push_back(m_imgLabel[i]);
		i++;
	}

	// train
	m_svm->train(feaData, ROW_SAMPLE, label);
	/*cv::Ptr<cv::ml::TrainData> features = cv::ml::TrainData::create(feaData, cv::ml::SampleTypes::ROW_SAMPLE, label);
	m_svm->trainAuto(features, 16, cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C),
		cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),
		cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF),
		cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE), true);*/

	//m_svmModelFilePath = "";
	//m_svm->save(m_svmModelFilePath);


	return 0;

}

long Classify::Predict(cv::Mat &img, string *imgClass)
{
	if (img.empty() || img.depth() != CV_8U)
	{
		return -1;
	}
	Mat feaValue;
	m_lbp.ComputeLBPFeatureVector_Rotation_Uniform(img, Size(CELL_SIZE, CELL_SIZE), feaValue);
	if (feaValue.empty())
		return -2;
	
	int predictResult = m_svm->predict(feaValue);

	*imgClass = m_imgClassName[predictResult];

	return 0;
}