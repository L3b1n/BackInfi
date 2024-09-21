#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelPPHumanSeg.h"

namespace BackInfi
{
    ModelPPHumanSeg::ModelPPHumanSeg()
    {
    }

    ModelPPHumanSeg::~ModelPPHumanSeg()
    {
    }

    void ModelPPHumanSeg::PrepareInputToNetwork(
		cv::Mat& resizedImage,
		cv::Mat& preprocessedImage)
	{
		BC_PROFILE_FUNC();

		resizedImage = (resizedImage / 256.0 - cv::Scalar(0.5, 0.5, 0.5))
			/ cv::Scalar(0.5, 0.5, 0.5);

		Utils::HwcToChw(resizedImage, preprocessedImage);
	}

    cv::Mat ModelPPHumanSeg::GetNetworkOutput(
		const Activation& activation,
		const std::vector<std::vector<int64_t>>& outputDims,
		std::vector<std::vector<float>>& outputTensorValues)
	{
		BC_PROFILE_FUNC();

		uint32_t outputWidth = (int)outputDims[0].at(2);
		uint32_t outputHeight = (int)outputDims[0].at(1);
		int32_t outputChannels = CV_32FC2;

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

    void ModelPPHumanSeg::PostprocessOutput(cv::Mat& outputImage)
	{
		BC_PROFILE_FUNC();
		
		// take 1st channel
		std::vector<cv::Mat> outputImageSplit;
		cv::split(outputImage, outputImageSplit);

		cv::normalize(outputImageSplit[1], outputImage, 1.0, 0.0, cv::NORM_MINMAX);
	}
}