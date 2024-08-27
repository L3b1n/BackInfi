#pragma once

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	class ModelSINET : public ModelBCHW
	{
	public:
		ModelSINET();
		~ModelSINET();

		virtual void PrepareInputToNetwork(
			cv::Mat& resizedImage,
			cv::Mat& preprocessedImage) override;
	};

}

