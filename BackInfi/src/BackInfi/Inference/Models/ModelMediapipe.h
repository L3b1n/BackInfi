#pragma once

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	class ModelMediapipe : public Model
	{
	public:
		ModelMediapipe();
		~ModelMediapipe();

		virtual cv::Mat GetNetworkOutput(
			const Activation& activation,
			const std::vector<std::vector<int64_t>>& outputDims,
			std::vector<std::vector<float>>& outputTensorValues) override;

		virtual void PostprocessOutput(cv::Mat& outputImage) override;
	};

}

