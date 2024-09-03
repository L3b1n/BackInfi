#include "bcpch.h"

#include "BackInfi/Inference/Models/Model.h"

namespace BackInfi
{

	// Model
	// -----------------------------------------------------------------
	Model::Model()
	{
	}

	void Model::PopulateInputOutputNames(
		const std::unique_ptr<Ort::Session>& session,
		std::vector<Ort::AllocatedStringPtr>& inputNames,
		std::vector<Ort::AllocatedStringPtr>& outputNames)
	{
		Ort::AllocatorWithDefaultOptions allocator;

		inputNames.clear();
		outputNames.clear();
		inputNames.push_back(session->GetInputNameAllocated(0, allocator));
		outputNames.push_back(session->GetOutputNameAllocated(0, allocator));
	}

	bool Model::PopulateInputOutputShapes(
		const std::unique_ptr<Ort::Session>& session,
		std::vector<std::vector<int64_t>>& inputDims,
		std::vector<std::vector<int64_t>>& outputDims)
	{
		// Assuming model only has one input and one output image

		inputDims.clear();
		outputDims.clear();

		inputDims.push_back(std::vector<int64_t>());
		outputDims.push_back(std::vector<int64_t>());

		// Get output shape
		const Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
		const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		outputDims[0] = outputTensorInfo.GetShape();

		// Fix any -1 values in outputDims to 1
		for (auto& i : outputDims[0])
		{
			if (i == -1)
				i = 1;
		}

		// Get input shape
		const Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
		const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		inputDims[0] = inputTensorInfo.GetShape();

		// Fix any -1 values in inputDims to 1
		for (auto& i : inputDims[0])
		{
			if (i == -1)
				i = 1;
		}

		if (inputDims[0].size() < 3 || outputDims[0].size() < 3)
		{
			BC_CORE_ERROR("Input or output tensor dims are < 3. input = {0}, output = {1}",
				(int)inputDims.size(), (int)outputDims.size());
			return false;
		}

		return true;
	}

	void Model::AllocateTensorBuffers(
		const std::vector<std::vector<int64_t>>& inputDims,
		const std::vector<std::vector<int64_t>>& outputDims,
		std::vector<std::vector<float>>& outputTensorValues,
		std::vector<std::vector<float>>& inputTensorValues,
		std::vector<Ort::Value>& inputTensor,
		std::vector<Ort::Value>& outputTensor)
	{
		// Assuming model only has one input and one output images
		outputTensorValues.clear();
		outputTensor.clear();

		inputTensorValues.clear();
		inputTensor.clear();

		Ort::MemoryInfo memoryInfo =
			Ort::MemoryInfo::CreateCpu(
				OrtAllocatorType::OrtDeviceAllocator,
				OrtMemType::OrtMemTypeDefault
			);

		// Allocate buffers and build input and output tensors
		for (size_t i = 0; i < inputDims.size(); i++)
		{
			inputTensorValues.push_back(
				std::vector<float>(Utils::VectorProduct(inputDims[i]), 0.0f)
			);
			BC_CORE_INFO("Allocated {0} sized float-array for input {1}",
				(int)inputTensorValues[i].size(), (int)i);
			inputTensor.push_back(
				Ort::Value::CreateTensor<float>(
					memoryInfo,
					inputTensorValues[i].data(),
					inputTensorValues[i].size(),
					inputDims[i].data(),
					inputDims[i].size()
				)
			);
		}

		for (size_t i = 0; i < outputDims.size(); i++)
		{
			outputTensorValues.push_back(
				std::vector<float>(Utils::VectorProduct(outputDims[i]), 0.0f)
			);
			BC_CORE_INFO("Allocated {0} sized float-array for output {1}",
				(int)outputTensorValues[i].size(), (int)i);
			outputTensor.push_back(
				Ort::Value::CreateTensor<float>(
					memoryInfo,
					outputTensorValues[i].data(),
					outputTensorValues[i].size(),
					outputDims[i].data(),
					outputDims[i].size()
				)
			);
		}
	}

	void Model::GetNetworkInputSize(
		const std::vector<std::vector<int64_t>>& inputDims,
		cv::Size& size)
	{
		// BHWC
		size.width  = (int)inputDims[0][2];
		size.height = (int)inputDims[0][1];
	}

	void Model::PrepareInputToNetwork(
		cv::Mat& resizedImage,
		cv::Mat& preprocessedImage)
	{
		preprocessedImage = resizedImage / 255.0;
	}

	void Model::PostprocessOutput(cv::Mat& output)
	{
	}

	void Model::LoadInputToTensor(
		const cv::Mat& preprocessedImage,
		cv::Size inputSize,
		std::vector<std::vector<float>>& inputTensorValues)
	{
		preprocessedImage.copyTo(
			cv::Mat(
				inputSize,
				CV_32FC3,
				&(inputTensorValues[0][0])
			)
		);
	}

	cv::Mat Model::GetNetworkOutput(
		const Activation& activation,
		const std::vector<std::vector<int64_t>>& outputDims,
		std::vector<std::vector<float>>& outputTensorValues)
	{
		// BHWC
		uint32_t outputWidth = (int)outputDims[0].at(2);
		uint32_t outputHeight = (int)outputDims[0].at(1);
		int32_t outputChannels =
			CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(3));

		return cv::Mat(outputHeight, outputWidth,
			outputChannels, outputTensorValues[0].data());
	}

	void Model::AssignOutputToInput(
		std::vector<std::vector<float>>&,
		std::vector<std::vector<float>>&)
	{
	}

	void Model::RunNetworkInference(
		const std::unique_ptr<Ort::Session>& session,
		const std::vector<Ort::AllocatedStringPtr>& inputNames,
		const std::vector<Ort::AllocatedStringPtr>& outputNames,
		const std::vector<Ort::Value>& inputTensor,
		std::vector<Ort::Value>& outputTensor)
	{
		if (inputNames.size() == 0 || outputNames.size() == 0 ||
			inputTensor.size() == 0 || outputTensor.size() == 0)
		{
			BC_CORE_ERROR("Error! Skip network inference. Inputs or outputs are null.");
			return;
		}

		std::vector<const char*> rawInputNames;
		for (auto& inputName : inputNames)
		{
			rawInputNames.push_back(inputName.get());
		}

		std::vector<const char*> rawOutputNames;
		for (auto& outputName : outputNames)
		{
			rawOutputNames.push_back(outputName.get());
		}

		session->Run(
			Ort::RunOptions{nullptr},
			rawInputNames.data(),
			inputTensor.data(),
			inputNames.size(),
			rawOutputNames.data(),
			outputTensor.data(),
			outputNames.size()
		);
	}

	// ModelBCHW
	// -----------------------------------------------------------------
	ModelBCHW::ModelBCHW()
	{
	}

	void ModelBCHW::PrepareInputToNetwork(
		cv::Mat& resizedImage,
		cv::Mat& preprocessedImage)
	{
		resizedImage = resizedImage / 255.0;
		Utils::HwcToChw(resizedImage, preprocessedImage);
	}

	void ModelBCHW::PostprocessOutput(cv::Mat& output)
	{
		cv::Mat outputTransposed;
		Utils::ChwToHwc32f(output, outputTransposed);
		outputTransposed.copyTo(output);
	}

	void ModelBCHW::GetNetworkInputSize(
		const std::vector<std::vector<int64_t>>& inputDims,
		cv::Size& size)
	{
		// BCHW
		size.width  = (int)inputDims[0][3];
		size.height = (int)inputDims[0][2];
	}

	cv::Mat ModelBCHW::GetNetworkOutput(
		const Activation& activation,
		const std::vector<std::vector<int64_t>>& outputDims,
		std::vector<std::vector<float>>& outputTensorValues)
	{
		// BCHW
		uint32_t outputWidth = (int)outputDims[0].at(3);
		uint32_t outputHeight = (int)outputDims[0].at(2);
		int32_t outputChannels = CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(1));

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

	void ModelBCHW::LoadInputToTensor(
		const cv::Mat& preprocessedImage,
		cv::Size,
		std::vector<std::vector<float>>& inputTensorValues)
	{
		inputTensorValues[0].assign(
			preprocessedImage.begin<float>(),
			preprocessedImage.end<float>()
		);
	}

}