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

	// 计算基本的256维LBP特征向量
	void ComputeLBPFeatureVector_256(const Mat &srcImage, Size cellSize, Mat &featureVector);
	void ComputeLBPImage_256(const Mat &srcImage, Mat &LBPImage);// 计算256维LBP特征图

																 // 计算灰度不变+等价模式LBP特征向量(58种模式)
	void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector);
	void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage);// 计算等价模式LBP特征图

																	 // 计算灰度不变+旋转不变+等价模式LBP特征向量(9种模式)
	void ComputeLBPFeatureVector_Rotation_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector);
	void ComputeLBPImage_Rotation_Uniform(const Mat &srcImage, Mat &LBPImage); // 计算灰度不变+旋转不变+等价模式LBP特征图,使用查找表

																			   // Test
	void Test();// 测试灰度不变+旋转不变+等价模式LBP
	void TestGetMinBinaryLUT();

private:
	void BuildUniformPatternTable(int *table); // 计算等价模式查找表
	int GetHopCount(int i);// 获取i中0,1的跳变次数

	void ComputeLBPImage_Rotation_Uniform_2(const Mat &srcImage, Mat &LBPImage);// 计算灰度不变+旋转不变+等价模式LBP特征图,不使用查找表
	int ComputeValue9(int value58); // 计算9种等价模式
	int GetMinBinary(int binary);// 通过LUT计算最小二进制
	uchar GetMinBinary(uchar *binary); // 计算得到最小二进制

};

#endif

