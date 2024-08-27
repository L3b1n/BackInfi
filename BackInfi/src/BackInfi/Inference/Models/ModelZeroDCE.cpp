#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelZeroDCE.h"

namespace BackInfi
{

	ModelZeroDCE::ModelZeroDCE()
	{
	}

	ModelZeroDCE::~ModelZeroDCE()
	{
	}

	void ModelZeroDCE::PostprocessOutput(cv::Mat& output)
	{
		// output is already BHWC and 0-255... nothing to do
	}

	cv::Mat ModelZeroDCE::GetNetworkOutput(
		const Activation& activation,
		const std::vector<std::vector<int64_t>>& outputDims,
		std::vector<std::vector<float>>& outputTensorValues)
	{
		// BHWC
		uint32_t outputWidth = (int)outputDims[0].at(1);
		uint32_t outputHeight = (int)outputDims[0].at(0);
		int32_t outputChannels = CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(2));

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

}
