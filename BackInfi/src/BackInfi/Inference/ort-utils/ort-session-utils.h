#pragma once

#include <opencv2/core/types.hpp>

#include "BackInfi/Inference/FilterData.h"

namespace BackInfi
{

	void createOrtSession(BackgroundRemovalFilter *tf);

	bool runFilterModelInference(BackgroundRemovalFilter *tf, const cv::Mat &imageBGRA, cv::Mat &output);

}

