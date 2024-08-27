#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelURetinex.h"

namespace BackInfi
{

	ModelURetinex::ModelURetinex()
	{
	}

	ModelURetinex::~ModelURetinex()
	{
	}

	void ModelURetinex::PopulateInputOutputNames(
		const std::unique_ptr<Ort::Session>& session,
		std::vector<Ort::AllocatedStringPtr>& inputNames,
		std::vector<Ort::AllocatedStringPtr>& outputNames)
	{
		Ort::AllocatorWithDefaultOptions allocator;

		inputNames.clear();
		outputNames.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++)
		{
			inputNames.push_back(session->GetInputNameAllocated(i, allocator));
		}
		for (size_t i = 0; i < session->GetOutputCount(); i++)
		{
			outputNames.push_back(session->GetOutputNameAllocated(i, allocator));
		}
	}

	bool ModelURetinex::PopulateInputOutputShapes(
		const std::unique_ptr<Ort::Session>& session,
		std::vector<std::vector<int64_t>>& inputDims,
		std::vector<std::vector<int64_t>>& outputDims)
	{
		// Assuming model only has one input and one output image
		inputDims.clear();
		outputDims.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++)
		{
			// Get input shape
			const Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
			const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
			inputDims.push_back(inputTensorInfo.GetShape());
		}

		for (size_t i = 0; i < session->GetOutputCount(); i++)
		{
			// Get output shape
			const Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
			const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
			outputDims.push_back(outputTensorInfo.GetShape());
		}

		return true;
	}

	void ModelURetinex::LoadInputToTensor(
		const cv::Mat& preprocessedImage,
		cv::Size,
		std::vector<std::vector<float>>& inputTensorValues)
	{
		inputTensorValues[0].assign(
			preprocessedImage.begin<float>(),
			preprocessedImage.end<float>()
		);
		inputTensorValues[1][0] = 5.0f;
	}

}
