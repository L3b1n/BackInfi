#pragma once

#include "BackInfi/Core/Base.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace BackInfi
{

	namespace Utils
	{
		template<typename T>
		T VectorProduct(const std::vector<T>& v)
		{
			T product = 1;
			for (auto& i : v)
			{
				// Turn 0 or -1, which are usually used as "None" (meaning any size), to 1s
				if (i > 0)
					product *= i;
			}
			return product;
		}

		static void HwcToChw(
			cv::InputArray src,
			cv::OutputArray dst)
		{
			std::vector<cv::Mat> channels;
			cv::split(src, channels);

			// Stretch one-channel images to vector
			for (auto& img : channels)
			{
				img = img.reshape(1, 1);
			}

			// Concatenate three vectors to one
			cv::hconcat(channels, dst);
		}

		/*
		* Convert a CHW Mat to HWC
		* Assume the input Mat is a 3D tensor of shape (C, H, W), but the Mat header has
		* the correct shape (H, W, C). This function will swap the channels to make it
		* (H, W, C) on the data level.
		*
		* @param src Input Mat, assume data is in CHW format, type is float32
		* @param dst Output Mat, data is in HWC format, type is float32
		*/
		static void ChwToHwc32f(
			cv::InputArray src,
			cv::OutputArray dst)
		{
			const cv::Mat srcMat = src.getMat();
			const int channels = srcMat.channels();
			const int height = srcMat.rows;
			const int width = srcMat.cols;
			const int dtype = srcMat.type();
			assert(dtype == CV_32F);
			const int channelStride = height * width;

			// Flatten to a vector of channels
			cv::Mat flatMat = srcMat.reshape(1, 1);

			std::vector<cv::Mat> channelsVec(channels);
			// Split the vector into channels
			for (int i = 0; i < channels; i++) {
				channelsVec[i] =
					cv::Mat(height, width, CV_MAKE_TYPE(dtype, 1), flatMat.ptr<float>(0) + i * channelStride);
			}

			cv::merge(channelsVec, dst);
		}
	}

	/*
	* Base class for all models
	*
	* Assume that all models have one input and one output.
	* The input is a 4D tensor of shape (1, H, W, C) where H and W are the height and width of
	* the input image.
	* The range of the input is [0, 255]. The input is in BGR format.
	* The input is a 32-bit floating point tensor.
	* This base model will convert the input to [0,1] and then the output to [0,255].
	*
	* Inheriting classes may override the methods for loading the model and running inference
	* with different pre-post processing behavior (like BCHW instead of BHWC or different ranges).
	*/
	class Model
	{
	public:
		const enum class Activation
		{
			NONE = 0,
			SIGMOID = 1,
			SOFTMAX = 2
		};

	public:
		Model();
		virtual ~Model() = default;

		virtual void PopulateInputOutputNames(
			const std::unique_ptr<Ort::Session>& session,
			std::vector<Ort::AllocatedStringPtr>& inputNames,
			std::vector<Ort::AllocatedStringPtr>& outputNames);

		virtual bool PopulateInputOutputShapes(
			const std::unique_ptr<Ort::Session>& session,
			std::vector<std::vector<int64_t>>& inputDims,
			std::vector<std::vector<int64_t>>& outputDims);

		virtual void AllocateTensorBuffers(
			const std::vector<std::vector<int64_t>>& inputDims,
			const std::vector<std::vector<int64_t>>& outputDims,
			std::vector<std::vector<float>>& outputTensorValues,
			std::vector<std::vector<float>>& inputTensorValues,
			std::vector<Ort::Value>& inputTensor,
			std::vector<Ort::Value>& outputTensor);

		virtual void GetNetworkInputSize(
			const std::vector<std::vector<int64_t>>& inputDims,
			cv::Size& size);

		virtual void PrepareInputToNetwork(
			cv::Mat& resizedImage,
			cv::Mat& preprocessedImage);

		/*
		* Postprocess the output of the network
		*
		* @param output The output of the network. This function should ensure the output is with
		* values in the range 0-1 (float 32), and in the BHWC format
		*/
		virtual void PostprocessOutput(cv::Mat& output);

		virtual void LoadInputToTensor(
			const cv::Mat& preprocessedImage,
			cv::Size inputSize,
			std::vector<std::vector<float>>& inputTensorValues);

		virtual cv::Mat GetNetworkOutput(
			const Activation& activation,
			const std::vector<std::vector<int64_t>>& outputDims,
			std::vector<std::vector<float>>& outputTensorValues);

		virtual void AssignOutputToInput(
			std::vector<std::vector<float>>&,
			std::vector<std::vector<float>>&);

		virtual void RunNetworkInference(
			const std::unique_ptr<Ort::Session>& session,
			const std::vector<Ort::AllocatedStringPtr>& inputNames,
			const std::vector<Ort::AllocatedStringPtr>& outputNames,
			const std::vector<Ort::Value>& inputTensor,
			std::vector<Ort::Value>& outputTensor);
	};

	class ModelBCHW : public Model
	{
	public:
		ModelBCHW();
		virtual ~ModelBCHW() = default;

		virtual void PrepareInputToNetwork(
			cv::Mat& resizedImage,
			cv::Mat& preprocessedImage) override;

		virtual void PostprocessOutput(cv::Mat& output) override;

		virtual void GetNetworkInputSize(
			const std::vector<std::vector<int64_t>>& inputDims,
			cv::Size& size) override;

		virtual cv::Mat GetNetworkOutput(
			const Activation& activation,
			const std::vector<std::vector<int64_t>>& outputDims,
			std::vector<std::vector<float>>& outputTensorValues) override;

		virtual void LoadInputToTensor(
			const cv::Mat& preprocessedImage,
			cv::Size,
			std::vector<std::vector<float>>& inputTensorValues) override;
	};

}
