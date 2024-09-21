#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelTBEFN.h"

namespace BackInfi
{
	ModelTBEFN::ModelTBEFN()
	{
	}

	ModelTBEFN::~ModelTBEFN()
	{
	}

	void ModelTBEFN::PostprocessOutput(cv::Mat& output)
	{
		BC_PROFILE_FUNC();

		// output is already BHWC ...
		output = output * 255.0; // Convert to 0-255 range
	}

	cv::Mat ModelTBEFN::GetNetworkOutput(
		const Activation& activation,
		const std::vector<std::vector<int64_t>>& outputDims,
		std::vector<std::vector<float>>& outputTensorValues)
	{
		BC_PROFILE_FUNC();
		
		// BHWC
		uint32_t outputWidth = (int)outputDims[0].at(2);
		uint32_t outputHeight = (int)outputDims[0].at(1);
		int32_t outputChannels = CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(3));

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

}