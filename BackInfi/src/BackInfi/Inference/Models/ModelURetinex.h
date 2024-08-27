#pragma once

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	class ModelURetinex : public ModelBCHW
	{
	public:
		ModelURetinex();
		~ModelURetinex();

		virtual void PopulateInputOutputNames(
			const std::unique_ptr<Ort::Session>& session,
			std::vector<Ort::AllocatedStringPtr>& inputNames,
			std::vector<Ort::AllocatedStringPtr>& outputNames) override;

		virtual bool PopulateInputOutputShapes(
			const std::unique_ptr<Ort::Session>& session,
			std::vector<std::vector<int64_t>>& inputDims,
			std::vector<std::vector<int64_t>>& outputDims) override;

		virtual void LoadInputToTensor(
			const cv::Mat& preprocessedImage,
			cv::Size,
			std::vector<std::vector<float>>& inputTensorValues) override;
	};

}

