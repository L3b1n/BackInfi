#pragma once

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	class ModelZeroDCE : public ModelBCHW
	{
	public:
		ModelZeroDCE();
		~ModelZeroDCE();

		virtual void PostprocessOutput(cv::Mat& output) override;

		virtual cv::Mat GetNetworkOutput(
			const Activation& activation,
			const std::vector<std::vector<int64_t>>& outputDims,
			std::vector<std::vector<float>>& outputTensorValues) override;
	};

}

