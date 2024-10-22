#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelRVM.h"

namespace BackInfi
{
	ModelRVM::ModelRVM()
	{
	}

	ModelRVM::~ModelRVM()
	{
	}

	void ModelRVM::PopulateInputOutputNames(
		const std::unique_ptr<Ort::Session>& session,
		std::vector<Ort::AllocatedStringPtr>& inputNames,
		std::vector<Ort::AllocatedStringPtr>& outputNames)
	{
		BC_PROFILE_FUNC();

		Ort::AllocatorWithDefaultOptions allocator;

		inputNames.clear();
		outputNames.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++)
		{
			inputNames.push_back(session->GetInputNameAllocated(i, allocator));
		}
		for (size_t i = 1; i < session->GetOutputCount(); i++)
		{
			outputNames.push_back(session->GetOutputNameAllocated(i, allocator));
		}
	}

	bool ModelRVM::PopulateInputOutputShapes(
		const std::unique_ptr<Ort::Session>& session,
		std::vector<std::vector<int64_t>>& inputDims,
		std::vector<std::vector<int64_t>>& outputDims)
	{
		BC_PROFILE_FUNC();

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

		for (size_t i = 1; i < session->GetOutputCount(); i++)
		{
			// Get output shape
			const Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
			const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
			outputDims.push_back(outputTensorInfo.GetShape());
		}

		inputDims[0][0] = 1;
		inputDims[0][2] = 192;
		inputDims[0][3] = 192;
		for (size_t i = 1; i < 5; i++)
		{
			inputDims[i][0] = 1;
			inputDims[i][1] = (i == 1) ? 16 : (i == 2) ? 20 : (i == 3) ? 40 : 64;
			inputDims[i][2] = 192 / (2 << (i - 1));
			inputDims[i][3] = 192 / (2 << (i - 1));
		}

		outputDims[0][0] = 1;
		outputDims[0][2] = 192;
		outputDims[0][3] = 192;
		for (size_t i = 1; i < 5; i++)
		{
			outputDims[i][0] = 1;
			outputDims[i][2] = 192 / (2 << (i - 1));
			outputDims[i][3] = 192 / (2 << (i - 1));
		}
		return true;
	}

	void ModelRVM::LoadInputToTensor(
		const cv::Mat& preprocessedImage,
		cv::Size,
		std::vector<std::vector<float>>& inputTensorValues)
	{
		BC_PROFILE_FUNC();

		inputTensorValues[0].assign(
			preprocessedImage.begin<float>(),
			preprocessedImage.end<float>()
		);
		inputTensorValues[5][0] = 1.0f;
	}

	void ModelRVM::AssignOutputToInput(
		std::vector<std::vector<float>>& outputTensorValues,
		std::vector<std::vector<float>>& inputTensorValues)
	{
		BC_PROFILE_FUNC();
		
		for (size_t i = 1; i < 5; i++)
		{
			inputTensorValues[i].assign(
				outputTensorValues[i].begin(),
				outputTensorValues[i].end()
			);
		}
	}

}
