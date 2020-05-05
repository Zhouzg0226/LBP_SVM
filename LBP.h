#pragma once

#ifndef __LBP_H__
#define __LBP_H__
#include "opencv2/opencv.hpp"
#include<vector>
using namespace std;
using namespace cv;


class LBP
{

public:

	// ���������256άLBP��������
	void ComputeLBPFeatureVector_256(const Mat &srcImage, Size cellSize, Mat &featureVector);
	void ComputeLBPImage_256(const Mat &srcImage, Mat &LBPImage);// ����256άLBP����ͼ

																 // ����ҶȲ���+�ȼ�ģʽLBP��������(58��ģʽ)
	void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector);
	void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage);// ����ȼ�ģʽLBP����ͼ

																	 // ����ҶȲ���+��ת����+�ȼ�ģʽLBP��������(9��ģʽ)
	void ComputeLBPFeatureVector_Rotation_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector);
	void ComputeLBPImage_Rotation_Uniform(const Mat &srcImage, Mat &LBPImage); // ����ҶȲ���+��ת����+�ȼ�ģʽLBP����ͼ,ʹ�ò��ұ�

																			   // Test
	void Test();// ���ԻҶȲ���+��ת����+�ȼ�ģʽLBP
	void TestGetMinBinaryLUT();

private:
	void BuildUniformPatternTable(int *table); // ����ȼ�ģʽ���ұ�
	int GetHopCount(int i);// ��ȡi��0,1���������

	void ComputeLBPImage_Rotation_Uniform_2(const Mat &srcImage, Mat &LBPImage);// ����ҶȲ���+��ת����+�ȼ�ģʽLBP����ͼ,��ʹ�ò��ұ�
	int ComputeValue9(int value58); // ����9�ֵȼ�ģʽ
	int GetMinBinary(int binary);// ͨ��LUT������С������
	uchar GetMinBinary(uchar *binary); // ����õ���С������

};

#endif

