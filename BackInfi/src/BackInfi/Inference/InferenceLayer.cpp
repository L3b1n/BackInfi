#include "bcpch.h"

#include "BackInfi/Inference/InferenceLayer.h"

#include "BackInfi/Inference/Models/ModelsWeights/MediapipeWeights.h"
//#include "BackInfi/Inference/Models/ModelsWeights/selfie_segmentation.h"
//#include "BackInfi/Inference/Models/ModelsWeights/SINet_Softmax_simple.h"
//#include "BackInfi/Inference/Models/ModelsWeights/rvm_mobilenetv3_fp32.h"
//#include "BackInfi/Inference/Models/ModelsWeights/pphumanseg_fp32.h"

namespace BackInfi
{

	namespace Utils
	{

		static void logger_fn(
			void* param,
			OrtLoggingLevel severity,
			const char* category,
			const char* logid,
			const char* code_location,
			const char* message)
		{
			switch (severity)
			{
			case ORT_LOGGING_LEVEL_VERBOSE:
				BC_CORE_TRACE("[Onnxruntime] VERB: {0} [{1}]", message, code_location);
				break;
			case ORT_LOGGING_LEVEL_INFO:
				BC_CORE_INFO("[Onnxruntime] INFO: {0} [{1}]", message, code_location);
				break;
			case ORT_LOGGING_LEVEL_WARNING:
				BC_CORE_WARN("[Onnxruntime] WARN: {0} [{1}]", message, code_location);
				break;
			case ORT_LOGGING_LEVEL_ERROR:
				BC_CORE_ERROR("[Onnxruntime] ERR: {0} [{1}]", message, code_location);
				break;
			case ORT_LOGGING_LEVEL_FATAL:
				BC_CORE_CRITICAL("[Onnxruntime] FATAL: {0} [{1}]", message, code_location);
				break;
			}
		}

	}

	InferenceLayer::InferenceLayer()
		: Layer("InferenceLayer")
	{
		m_Filter = nullptr;
	}

	void InferenceLayer::OnAttach()
	{
		m_Filter                 = std::make_unique<BackgroundRemoval1>();
		m_Filter->ModelSelection = MODEL_MEDIAPIPE;

		std::string instanceName{"background-removal-inference"};
		m_Filter->OnnxEnv.reset(
			new Ort::Env(
				OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
				instanceName.c_str(),
				Utils::logger_fn,
				nullptr
			)
		);
	}

	void InferenceLayer::OnDetach()
	{
		BC_CORE_TRACE("Excit inference layer!");
	}

	void InferenceLayer::OnUpdate(TimeStep ts)
	{
		UpdateFilter();

		//if (m_Filter->IsDisabled)
		//{
		//	BC_CORE_WARN("Error! Rendering of background filter is disabled!");
		//	return;
		//}
		if (m_Filter->OnnxSession.get() == nullptr)
		{
			BC_CORE_ERROR("Onnxruntime session isn't initialized!");
		}
		if (m_Filter->Model.get() == nullptr)
		{
			BC_CORE_ERROR("Model object isn't initialized!");
		}
		if (m_InputImage.data == nullptr)
		{
			BC_CORE_ERROR("Error! Input image is empty!");
			return;
		}

		cv::Size maskSize;
		m_Filter->Model->GetNetworkInputSize(
			m_Filter->InputDims,
			maskSize
		);

		// Resize to network input size before image similarity to increase
		// speed of filterVideoTick and overall fps
		cv::Mat resizedImageBGR;
		cv::resize(
			m_InputImage,
			resizedImageBGR,
			maskSize,
			cv::INTER_CUBIC
		);

		// Implement image similarity. If image is almost the same as 
		// the previous one, skip this frame and use previous mask
		if (m_Filter->EnableImageSimilarity)
		{
			if (!m_Filter->LastImage.empty() && !resizedImageBGR.empty() &&
				m_Filter->LastImage.size() == resizedImageBGR.size())
			{
				// Calculate PSNR
				double psnr = cv::PSNR(m_Filter->LastImage, resizedImageBGR);

				if (psnr > m_Filter->ImageSimilarityThreshold)
				{
					// The image is almost the same as the previous one. Skip processing.
					return;
				}
			}
			m_Filter->LastImage = resizedImageBGR.clone();
		}

		m_Filter->MaskEveryXFramesCount++;
		m_Filter->MaskEveryXFramesCount %= m_Filter->MaskEveryXFrames;

		try
		{
			if (m_Filter->MaskEveryXFramesCount != 0 && !m_BackgroundMask.empty())
			{
				// We are skipping processing of the mask for this frame.
				// Get the background mask previously generated.
				; // Do nothing
			}
			else
			{
				// Converting input image to the format and size suitable
				// for model to reduce coping massive cv matrix more times
				// To RGB.
				cv::Mat resizedImageRGB;
				cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::COLOR_BGR2RGB);

				// Prepare input to nework.
				cv::Mat resizedImage;
				resizedImageRGB.convertTo(resizedImage, CV_32F);

				// Process the image to find the mask.
				cv::Mat preprocessedImage;

				m_Filter->Model->PrepareInputToNetwork(resizedImage, preprocessedImage);

				m_Filter->Model->LoadInputToTensor(
					preprocessedImage,
					maskSize,
					m_Filter->InputTensorValues
				);

				// Run network inference
				m_Filter->Model->RunNetworkInference(
					m_Filter->OnnxSession,
					m_Filter->InputNames, m_Filter->OutputNames,
					m_Filter->InputTensor, m_Filter->OutputTensor
				);

				// Get output
				// Map network output to cv::Mat
				m_BackgroundMask = m_Filter->Model->GetNetworkOutput(
					m_Filter->UseFloatMask ? Model::Activation::SOFTMAX : Model::Activation::NONE,
					m_Filter->OutputDims,
					m_Filter->OutputTensorValues
				);

				if (!m_Filter->UseFloatMask)
				{
					// Post-process output. 
					// The image will now be in [0,1] float, BHWC format.
					m_Filter->Model->PostprocessOutput(m_BackgroundMask);

					// Convert [0,1] float to CV_8U [0,255]
					m_BackgroundMask.convertTo(m_BackgroundMask, CV_8U, 255.0);

					// Assume outputImage is now a single channel, 
					// uint8 image with values between 0 and 255

					// If have a threshold, apply it.
					// Otherwise, just use the output image as the mask
					if (m_Filter->EnableThreshold)
					{
						// It's needed to make m_Filter->threshold (float [0,1]) be in that range
						const uint8_t threshold_value =
							static_cast<uint8_t>(m_Filter->Threshold * 255.0f);
						m_BackgroundMask = m_BackgroundMask < threshold_value;
					}
					else
					{
						m_BackgroundMask = 255 - m_BackgroundMask;
					}
				}

				// Temporal smoothing.
				if (!m_Filter->LastBackgroundMask.empty() &&
					m_Filter->LastBackgroundMask.size() ==
					m_BackgroundMask.size())
				{
					float temporalSmoothFactor = m_Filter->TemporalSmoothFactor;
					if (m_Filter->EnableThreshold)
					{
						// The temporal smooth factor can't 
						// be smaller than the threshold
						temporalSmoothFactor =
							std::max(temporalSmoothFactor,
								m_Filter->Threshold);
					}

					cv::addWeighted(
						m_BackgroundMask,
						temporalSmoothFactor,
						m_Filter->LastBackgroundMask,
						1.0 - temporalSmoothFactor,
						0.0,
						m_BackgroundMask
					);
				}

				m_Filter->LastBackgroundMask = m_BackgroundMask.clone();

				// Contour processing
				// Only applicable if we are thresholding (and get a binary image)
				if (m_Filter->EnableThreshold)
				{
					if (!m_Filter->UseFloatMask &&
						m_Filter->ContourFilter > 0.0 && m_Filter->ContourFilter < 1.0)
					{
						std::vector<std::vector<cv::Point>> contours;
						cv::findContours(m_BackgroundMask, contours,
							cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
						std::vector<std::vector<cv::Point>> filteredContours;
						const int64_t contourSizeThreshold =
							static_cast<int64_t>(m_BackgroundMask.total() * m_Filter->ContourFilter);
						for (auto& contour : contours)
						{
							if (cv::contourArea(contour) > contourSizeThreshold)
								filteredContours.push_back(contour);
						}
						m_BackgroundMask.setTo(0);
						cv::drawContours(
							m_BackgroundMask,
							filteredContours,
							-1,
							cv::Scalar(255),
							-1
						);
					}

					if (m_Filter->SmoothContour > 0.0)
					{
						int k_size = (int)(3 + 11 * m_Filter->SmoothContour);
						k_size += k_size % 2 == 0 ? 1 : 0;
						cv::stackBlur(m_BackgroundMask, m_BackgroundMask, cv::Size(k_size, k_size));
					}

					// Additional contour processing at full resolution
					if (m_Filter->SmoothContour > 0.0 && !m_Filter->UseFloatMask)
					{
						// If the mask was smoothed, apply a Threshold to get a binary mask
						m_BackgroundMask = m_BackgroundMask > 128;
					}

					if (m_Filter->Feather > 0.0 && !m_Filter->UseFloatMask)
					{
						// Feather (blur) the mask
						int k_size = (int)(40 * m_Filter->Feather);
						k_size += k_size % 2 == 0 ? 1 : 0;
						cv::dilate(m_BackgroundMask, m_BackgroundMask,
							cv::Mat(), cv::Point(-1, -1), k_size / 3);
						cv::boxFilter(
							m_BackgroundMask,
							m_BackgroundMask,
							m_BackgroundMask.depth(),
							cv::Size(k_size, k_size)
						);
					}
				}

				cv::flip(m_BackgroundMask, m_BackgroundMask, 0);
			}
		}
		catch (const Ort::Exception& e)
		{
			BC_CORE_ASSERT(false, "Error! {0}", e.what());
			// TODO: Fall back to CPU if it makes sense
		}
		catch (const std::exception& e)
		{
			BC_CORE_ASSERT(false, "Error! {0}", e.what());
		}
	}

	void InferenceLayer::OnEvent(Event& event)
	{
		BC_CORE_TRACE("Inference layer on event.");
	}

	void InferenceLayer::UpdateSettings(const InferenceSettings& settings)
	{
		m_Filter->EnableThreshold = settings.EnableThreshold;
		m_Filter->Threshold = settings.Threshold;

		m_Filter->EnableImageSimilarity = settings.EnableImageSimilarity;
		m_Filter->ImageSimilarityThreshold = settings.ImageSimilarityThreshold;

		m_Filter->UseFloatMask = settings.UseFloatMask;

		m_Filter->ContourFilter = settings.ContourFilter;
		m_Filter->SmoothContour = settings.SmoothContour;
		m_Filter->Feather = settings.Feather;
		m_Filter->TemporalSmoothFactor = settings.TemporalSmoothFactor;

		m_Filter->MaskEveryXFrames = settings.MaskEveryXFrames;
		m_Filter->MaskEveryXFramesCount = (int)(0);

		m_Settings = settings;
	}

	void InferenceLayer::UpdateFilter()
	{
		if (m_Filter->ModelSelection.empty() || m_Filter->ModelSelection != m_Settings.Model ||
			m_Filter->UseGPU != m_Settings.UseGpu || m_Filter->NumThreads != m_Settings.NumThreads)
		{
			// Re-initialize Model if it's not already the selected one or switching inference device
			m_Filter->ModelSelection = m_Settings.Model;
			m_Filter->UseGPU         = m_Settings.UseGpu;
			m_Filter->NumThreads     = m_Settings.NumThreads;

			if (m_Filter->ModelSelection == MODEL_SINET)
			{
				m_Filter->Model.reset(new ModelSINET);
			}
			if (m_Filter->ModelSelection == MODEL_SELFIE)
			{
				m_Filter->Model.reset(new ModelSelfie);
			}
			if (m_Filter->ModelSelection == MODEL_MEDIAPIPE)
			{
				m_Filter->Model.reset(new ModelMediapipe);
			}
			if (m_Filter->ModelSelection == MODEL_RVM) {
				m_Filter->Model.reset(new ModelRVM);
			}
			if (m_Filter->ModelSelection == MODEL_PPHUMANSEG)
			{
				m_Filter->Model.reset(new ModelPPHumanSeg);
			}

			// Create Ort session
			if (m_Filter->Model.get() == nullptr)
			{
				BC_CORE_ASSERT(false, "Error! Model object is not initialized!");
				return;
			}

			Ort::SessionOptions sessionOptions;

			sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			if (m_Filter->UseGPU != USEGPU_CPU)
			{
				sessionOptions.DisableMemPattern();
				sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
			}
			else
			{
				sessionOptions.SetInterOpNumThreads(m_Filter->NumThreads);
				sessionOptions.SetIntraOpNumThreads(m_Filter->NumThreads);
			}

			if (m_Filter->ModelSelection.c_str() == nullptr)
			{
				BC_CORE_ERROR("Error! Unable to get model filename {0} from plugin!", m_Filter->ModelSelection.c_str());
				return;
			}

			uint32_t modelSize;
			const BYTE* modelWeights;
			//if (m_Filter->ModelSelection == MODEL_SINET)
			//{
			//    modelWeights = SINet_Softmax_simple_onnx;
			//    modelSize    = SINet_Softmax_simple_onnx_len;
			//}
			//if (m_Filter->ModelSelection == MODEL_SELFIE)
			//{
			//    modelWeights = selfie_segmentation_onnx;
			//    modelSize    = selfie_segmentation_onnx_len;
			//}
			if (m_Filter->ModelSelection == MODEL_MEDIAPIPE)
			{
				modelWeights = mediapipe_onnx;
				modelSize    = mediapipe_onnx_len;
			}
			//if (m_Filter->ModelSelection == MODEL_RVM)
			//{
			//    modelWeights = rvm_mobilenetv3_fp32_onnx;
			//    modelSize    = rvm_mobilenetv3_fp32_onnx_len;
			//}
			//if (m_Filter->ModelSelection == MODEL_PPHUMANSEG)
			//{
			//    modelWeights = pphumanseg_fp32_onnx;
			//    modelSize    = pphumanseg_fp32_onnx_len;
			//}

			BC_CORE_ASSERT(modelWeights && modelSize != 0, "Unable to get model {0} from plugin!", m_Filter->modelSelection);

			try
			{
				#ifdef BC_PLATFORM_LINUX
				if (m_Filter->useGPU == USEGPU_TENSORRT)
				{
					Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
				}
				#elif defined(BC_PLATFORM_WINDOWS)
				if (m_Filter->UseGPU == USEGPU_DML)
				{
					#if 0
					if (m_Filter->useGPU == USEGPU_DML)
					{
						auto& api = Ort::GetApi();
						OrtDmlApi* dmlApi = nullptr;
						Ort::ThrowOnError(
							api.GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dmlApi));
						Ort::ThrowOnError(
							dmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));
					}
					#endif
				}
				#elif defined(BC_PLATFORM_MACOS)
				if (m_Filter->useGPU == USEGPU_COREML)
				{
					uint32_t coreml_flags = 0;
					coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
					Ort::ThrowOnError(
						OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreml_flags));
				}
				#endif
				//std::string str = "C:\\Users\\leonid\\source\\repos\\OBSPlaginPortraitSegmentation\\x64\\Release\\models\\SINet_Softmax_simple.onnx";
				//std::wstring modelFilepath_ws(str.size(), L' ');
				//std::copy(str.begin(), str.end(), modelFilepath_ws.begin());
				//m_Filter->session.reset(new Ort::Session(*m_Filter->env, modelFilepath_ws.c_str(), sessionOptions));

				m_Filter->OnnxSession.reset(
					new Ort::Session(
						*m_Filter->OnnxEnv,
						modelWeights,
						modelSize,
						sessionOptions
					)
				);
			}
			catch (const std::exception& e)
			{
				BC_CORE_ERROR("Error! {0}", e.what());
				return;
			}

			Ort::AllocatorWithDefaultOptions allocator;

			m_Filter->Model->PopulateInputOutputNames(
				m_Filter->OnnxSession,
				m_Filter->InputNames,
				m_Filter->OutputNames
			);

			if (!m_Filter->Model->PopulateInputOutputShapes(
				m_Filter->OnnxSession, m_Filter->InputDims, m_Filter->OutputDims))
			{
				BC_CORE_ERROR("Error! Unable to get model input and output shapes!");
				return;
			}

			// Allocate buffers
			m_Filter->Model->AllocateTensorBuffers(
				m_Filter->InputDims,
				m_Filter->OutputDims,
				m_Filter->OutputTensorValues,
				m_Filter->InputTensorValues,
				m_Filter->InputTensor,
				m_Filter->OutputTensor
			);
		}
	}

}