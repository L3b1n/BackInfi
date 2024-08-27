#pragma once

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	class ModelRVM : public ModelBCHW
	{
	public:
		ModelRVM();
		~ModelRVM();

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

		virtual void AssignOutputToInput(
			std::vector<std::vector<float>>& outputTensorValues,
			std::vector<std::vector<float>>& inputTensorValues) override;
	};

}

