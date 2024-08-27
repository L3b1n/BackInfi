#pragma once

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	class ModelSelfie : public Model
	{
	public:
		ModelSelfie();
		~ModelSelfie();

		virtual void PostprocessOutput(cv::Mat& outputImage) override;
	};

}

