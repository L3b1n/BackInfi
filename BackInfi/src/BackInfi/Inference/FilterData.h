#pragma once

#include "BackInfi/Inference/Models/Model.h"
#include "BackInfi/Inference/ort-utils/ORTModelData.h"

namespace BackInfi
{

	/**
	  * @brief The filter_data struct
	  *
	  * This struct is used to store the base data needed for ORT filters.
	  *
	*/
	struct FilterData : public ORTModelData
	{
		std::string useGPU;
		uint32_t numThreads;
		std::string modelSelection;
		std::unique_ptr<Model> model;

		cv::Mat inputRGBA;

		bool isDisabled;

		std::mutex inputRGBALock;
		std::mutex outputLock;

		const unsigned char* modelInfo = nullptr;
		unsigned int modelSize = 0;
	};

	struct BackgroundRemovalFilter : public FilterData
	{
		bool enableThreshold = true;
		float threshold = 0.5f;
		cv::Scalar backgroundColor{0, 0, 0, 0};
		float contourFilter = 0.05f;
		float smoothContour = 0.5f;
		float feather = 0.0f;

		cv::Mat backgroundMask;
		int maskEveryXFrames = 1;
		int maskEveryXFramesCount = 0;
		int64_t blurBackground = 0;
	};

}
