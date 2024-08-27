#pragma once

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	class ModelPPHumanSeg : public ModelBCHW
	{
	public:
		ModelPPHumanSeg();
		~ModelPPHumanSeg();

		virtual void PrepareInputToNetwork(
			cv::Mat& resizedImage,
			cv::Mat& preprocessedImage) override;

		virtual cv::Mat GetNetworkOutput(
			const Activation& activation,
			const std::vector<std::vector<int64_t>>& outputDims,
			std::vector<std::vector<float>>& outputTensorValues) override;

		virtual void PostprocessOutput(cv::Mat& outputImage) override;
	};

}
