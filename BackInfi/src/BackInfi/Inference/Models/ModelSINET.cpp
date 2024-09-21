#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelSINET.h"

namespace BackInfi
{
	ModelSINET::ModelSINET()
	{
	}

	ModelSINET::~ModelSINET()
	{
	}

	void ModelSINET::PrepareInputToNetwork(
		cv::Mat& resizedImage,
		cv::Mat& preprocessedImage)
	{
		BC_PROFILE_FUNC();
		
		cv::subtract(
			resizedImage, 
			cv::Scalar(102.890434, 111.25247, 126.91212),
			resizedImage
		);
		cv::multiply(
			resizedImage,
			cv::Scalar(1.0 / 62.93292, 1.0 / 62.82138, 1.0 / 66.355705) / 255.0,
			resizedImage
		);
		Utils::HwcToChw(resizedImage, preprocessedImage);
	}

}