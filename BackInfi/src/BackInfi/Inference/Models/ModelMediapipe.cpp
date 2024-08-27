#include "bcpch.h"

#include "BackInfi/Inference/Models/ModelMediapipe.h"

namespace BackInfi
{

	namespace Utils
	{
		template <class T>
		void ApplyActivation(
			cv::Mat& tensor_mat,
			cv::Mat* small_mask_mat,
			Model::Activation activation = Model::Activation::SOFTMAX)
		{
			// Configure activation function.
			const int output_layer_index = 0;
			const auto activation_fn = [&](const cv::Vec2f& mask_value) {
				float new_mask_value = 0;
				// TODO consider moving switch out of the loop,
				// and also avoid float/Vec2f casting.
				switch (activation)
				{
				case Model::Activation::NONE: { //NONE
					new_mask_value = mask_value[0];
					break;
				}
				case Model::Activation::SIGMOID: { //SIGMOID
					const float pixel0 = mask_value[0];
					new_mask_value =
						static_cast<float>(1.0 / (std::exp(-pixel0) + 1.0));
					break;
				}
				case Model::Activation::SOFTMAX: { //SOFTMAX
					const float pixel0 = mask_value[0];
					const float pixel1 = mask_value[1];
					const float max_pixel = max(pixel0, pixel1);
					const float min_pixel = min(pixel0, pixel1);
					const float softmax_denom =
						/*exp(max_pixel - max_pixel)=*/1.0f +
						std::exp(min_pixel - max_pixel);
					new_mask_value = std::exp(mask_value[output_layer_index] - max_pixel) /
						softmax_denom;
					break;
				}
				}
				return new_mask_value;
			};

			// Process mask tensor.
			for (int i = 0; i < tensor_mat.rows; ++i)
			{
				for (int j = 0; j < tensor_mat.cols; ++j)
				{
					const T& input_pix = tensor_mat.at<T>(i, j);
					const float mask_value = activation_fn(input_pix);
					small_mask_mat->at<float>(i, j) = mask_value;
				}
			}
		}
	}

	ModelMediapipe::ModelMediapipe()
	{
	}

	ModelMediapipe::~ModelMediapipe()
	{
	}

	cv::Mat ModelMediapipe::GetNetworkOutput(
		const Activation& activation,
		const std::vector<std::vector<int64_t>>& outputDims,
		std::vector<std::vector<float>>& outputTensorValues)
	{
		if (activation == Activation::NONE)
		{
			// Basic activation
			uint32_t outputWidth = (int)outputDims[0].at(2);
			uint32_t outputHeight = (int)outputDims[0].at(1);
			int32_t outputChannels = CV_32FC2;

			return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
		}
		else if (activation == Activation::SOFTMAX)
		{
			// Softmax activation
			uint32_t tensor_width = (int)outputDims[0].at(2);
			uint32_t tensor_height = (int)outputDims[0].at(1);
			int32_t tensor_channels = 2;

			// Create initial working mask.
			cv::Mat small_mask_mat(cv::Size(tensor_width, tensor_height), CV_32FC1);

			// Wrap input tensor.
			auto raw_input_tensor = &outputTensorValues[0];
			const float* raw_input_data = raw_input_tensor->data();
			cv::Mat tensor_mat(
				cv::Size(tensor_width, tensor_height),
				CV_MAKETYPE(CV_32F, tensor_channels),
				const_cast<float*>(raw_input_data)
			);

			// Process mask tensor and apply activation function.
			Utils::ApplyActivation<cv::Vec2f>(tensor_mat, &small_mask_mat, activation);

			return small_mask_mat;
		}

		BC_CORE_ASSERT(false, "Provided unsuported activation for this model!");
		return cv::Mat();
	}

	void ModelMediapipe::PostprocessOutput(cv::Mat& outputImage)
	{
		// Take 2nd channel
		std::vector<cv::Mat> outputImageSplit;
		cv::split(outputImage, outputImageSplit);
		outputImage = outputImageSplit[outputImageSplit.size() == 2 ? 1 : 0];
	}

}