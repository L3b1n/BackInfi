#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelSelfie.h"

namespace BackInfi
{

	ModelSelfie::ModelSelfie()
	{
	}

	ModelSelfie::~ModelSelfie()
	{
	}

	void ModelSelfie::PostprocessOutput(cv::Mat& outputImage)
	{
		cv::normalize(outputImage, outputImage, 1.0, 0.0, cv::NORM_MINMAX);
	}

}