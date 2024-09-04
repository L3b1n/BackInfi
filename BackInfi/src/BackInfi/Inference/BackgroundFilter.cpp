#include "bcpch.h"

#include "BackInfi/Inference/BackgroundFilter.h"

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

	BackgroundFilter::BackgroundFilter()
	{
		m_Width  = 0;
		m_Height = 0;
	}

	BackgroundFilter::~BackgroundFilter()
	{
		m_Width  = 0;
		m_Height = 0;
	}

	void BackgroundFilter::SetUp(const int& height, const int& width, const bool& blur)
	{
		m_Filter                 = std::make_unique<BackgroundRemoval>();
		m_Width                  = width;
		m_Height                 = height;
		m_Blur                   = blur;
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

	void BackgroundFilter::LoadBackground(const std::string path)
	{
		std::string backgroundImagePath = std::filesystem::current_path().append(path).string();
		m_Background = cv::imread(backgroundImagePath);
		cv::resize(m_Background, m_Background, cv::Size(m_Width, m_Height));
		cv::flip(m_Background, m_Background, 0);
	}

	void BackgroundFilter::FilterUpdate(const Settings& settings)
	{
		m_Filter->EnableThreshold          = settings.EnableThreshold;
		m_Filter->Threshold                = settings.Threshold;

		m_Filter->EnableImageSimilarity    = settings.EnableImageSimilarity;
		m_Filter->ImageSimilarityThreshold = settings.ImageSimilarityThreshold;

		m_Filter->UseFloatMask             = settings.UseFloatMask;

		m_Filter->ContourFilter            = settings.ContourFilter;
		m_Filter->SmoothContour            = settings.SmoothContour;
		m_Filter->Feather                  = settings.Feather;
		m_Filter->BlurBackground           = settings.BlurBackground;
		m_Filter->TemporalSmoothFactor     = settings.TemporalSmoothFactor;

		m_Filter->MaskEveryXFrames         = settings.MaskEveryXFrames;
		m_Filter->MaskEveryXFramesCount    = (int)(0);

		const std::string newUseGpu        = settings.UseGpu;
		const std::string newModel         = settings.Model;
		const uint32_t newNumThreads       = settings.NumThreads;

		if (m_Filter->ModelSelection.empty() || m_Filter->ModelSelection != newModel ||
			m_Filter->UseGPU != newUseGpu || m_Filter->NumThreads != newNumThreads)
		{
			// Re-initialize Model if it's not already the selected one or switching inference device
			m_Filter->ModelSelection = newModel;
			m_Filter->UseGPU = newUseGpu;
			m_Filter->NumThreads = newNumThreads;

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
				modelSize = mediapipe_onnx_len;
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

			try {
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

	void BackgroundFilter::FilterVideoTick(const cv::Mat& imageBGR)
	{
		if (m_Filter->IsDisabled)
		{
			BC_CORE_WARN("Error! Rendering of background filter is disabled!");
			return;
		}
		if (m_Filter->OnnxSession.get() == nullptr)
		{
			BC_CORE_ERROR("Onnxruntime session isn't initialized!");
		}
		if (m_Filter->Model.get() == nullptr)
		{
			BC_CORE_ERROR("Model object isn't initialized!");
		}
		if (imageBGR.data == nullptr)
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
			imageBGR,
			resizedImageBGR,
			maskSize,
			cv::INTER_CUBIC
		);

		// If BackInfi in blur background mode, save resized input image
		// for the blending part
		if (m_Blur)
		{
			m_Background = resizedImageBGR;
		}

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
				cv::Mat backgroundMask, preprocessedImage;

				m_Filter->Model->PrepareInputToNetwork(resizedImage, preprocessedImage);

				m_Filter->Model->LoadInputToTensor(
					preprocessedImage,
					maskSize,
					m_Filter->InputTensorValues
				);

				// Run network inference
				m_Filter->Model->RunNetworkInference(
					m_Filter->OnnxSession,
					m_Filter->InputNames,  m_Filter->OutputNames,
					m_Filter->InputTensor, m_Filter->OutputTensor
				);

				// Get output
				// Map network output to cv::Mat
				backgroundMask = m_Filter->Model->GetNetworkOutput(
					m_Filter->UseFloatMask ? Model::Activation::SOFTMAX : Model::Activation::NONE,
					m_Filter->OutputDims,
					m_Filter->OutputTensorValues
				);

				if (!m_Filter->UseFloatMask)
				{
					// Post-process output. 
					// The image will now be in [0,1] float, BHWC format.
					m_Filter->Model->PostprocessOutput(backgroundMask);

					// Convert [0,1] float to CV_8U [0,255]
					backgroundMask.convertTo(backgroundMask, CV_8U, 255.0);

					// Assume outputImage is now a single channel, 
					// uint8 image with values between 0 and 255

					// If have a threshold, apply it.
					// Otherwise, just use the output image as the mask
					if (m_Filter->EnableThreshold)
					{
						// It's needed to make m_Filter->threshold (float [0,1]) be in that range
						const uint8_t threshold_value =
							static_cast<uint8_t>(m_Filter->Threshold * 255.0f);
						backgroundMask = backgroundMask < threshold_value;
					}
					else
					{
						backgroundMask = 255 - backgroundMask;
					}
				}

				// Temporal smoothing.
				if (!m_Filter->LastBackgroundMask.empty() &&
					m_Filter->LastBackgroundMask.size() ==
					backgroundMask.size())
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
						backgroundMask,
						temporalSmoothFactor,
						m_Filter->LastBackgroundMask,
						1.0 - temporalSmoothFactor,
						0.0,
						backgroundMask
					);
				}

				m_Filter->LastBackgroundMask = backgroundMask.clone();

				// Contour processing
				// Only applicable if we are thresholding (and get a binary image)
				if (m_Filter->EnableThreshold)
				{
					if (!m_Filter->UseFloatMask &&
						m_Filter->ContourFilter > 0.0 && m_Filter->ContourFilter < 1.0)
					{
						std::vector<std::vector<cv::Point>> contours;
						cv::findContours(backgroundMask, contours,
							cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
						std::vector<std::vector<cv::Point>> filteredContours;
						const int64_t contourSizeThreshold =
							static_cast<int64_t>(backgroundMask.total() * m_Filter->ContourFilter);
						for (auto& contour : contours)
						{
							if (cv::contourArea(contour) > contourSizeThreshold)
								filteredContours.push_back(contour);
						}
						backgroundMask.setTo(0);
						cv::drawContours(
							backgroundMask,
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
						cv::stackBlur(backgroundMask, backgroundMask, cv::Size(k_size, k_size));
					}

					// Additional contour processing at full resolution
					if (m_Filter->SmoothContour > 0.0 && !m_Filter->UseFloatMask)
					{
						// If the mask was smoothed, apply a Threshold to get a binary mask
						backgroundMask = backgroundMask > 128;
					}

					if (m_Filter->Feather > 0.0 && !m_Filter->UseFloatMask)
					{
						// Feather (blur) the mask
						int k_size = (int)(40 * m_Filter->Feather);
						k_size += k_size % 2 == 0 ? 1 : 0;
						cv::dilate(backgroundMask, backgroundMask,
							cv::Mat(), cv::Point(-1, -1), k_size / 3);
						cv::boxFilter(
							backgroundMask,
							backgroundMask,
							m_BackgroundMask.depth(),
							cv::Size(k_size, k_size)
						);
					}
				}

				cv::flip(backgroundMask, backgroundMask, 0);

				m_BackgroundMask = backgroundMask.clone();
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

	void BackgroundFilter::BlendSegmentationSmoothing(
		const double combine_with_previous_ratio_)
	{
		if (m_BackgroundMask.empty())
		{ 
			BC_CORE_ASSERT(false, "Error! Background mask is empty!");
		}

		if (m_PreviousMask.empty())
			m_PreviousMask = m_BackgroundMask;

		if (m_PreviousMask.type() != m_BackgroundMask.type())
		{
			BC_CORE_WARN("Warning: mixing input format types: {0} != {1}", m_PreviousMask.type(), m_BackgroundMask.type());
			return;
		}

		if (m_PreviousMask.rows != m_BackgroundMask.rows)
			return;
		if (m_PreviousMask.cols != m_BackgroundMask.cols)
			return;

		// Setup destination image.
		cv::Mat output_mat(m_BackgroundMask.rows, m_BackgroundMask.cols, m_BackgroundMask.type());
		output_mat.setTo(cv::Scalar(0));

		// Blending function.
		const auto blending_fn = [&](const float prev_mask_value,
			const float new_mask_value) {
				/*
				 * Assume p := new_mask_value
				 * H(p) := 1 + (p * log(p) + (1-p) * log(1-p)) / log(2)
				 * uncertainty alpha(p) =
				 *   Clamp(1 - (1 - H(p)) * (1 - H(p)), 0, 1) [squaring the uncertainty]
				 *
				 * The following polynomial approximates uncertainty alpha as a function
				 * of (p + 0.5):
				 */
				const float c1 = 5.68842;
				const float c2 = -0.748699;
				const float c3 = -57.8051;
				const float c4 = 291.309;
				const float c5 = -624.717;
				const float t  = new_mask_value - 0.5f;
				const float x  = t * t;

				const float uncertainty =
					1.0f -
					std::min(1.0f, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))));

				return new_mask_value + (prev_mask_value - new_mask_value) *
					(uncertainty * combine_with_previous_ratio_);
		};

		// Write directly to the first channel of output.
		for (int i = 0; i < output_mat.rows; ++i)
		{
			float* out_ptr        = output_mat.ptr<float>(i);
			const float* curr_ptr = m_BackgroundMask.ptr<float>(i);
			const float* prev_ptr = m_PreviousMask.ptr<float>(i);
			for (int j = 0; j < output_mat.cols; ++j)
			{
				const float new_mask_value  = curr_ptr[j];
				const float prev_mask_value = prev_ptr[j];
				out_ptr[j]                  = blending_fn(prev_mask_value, new_mask_value);
			}
		}

		m_BackgroundMask = output_mat;
		m_PreviousMask   = output_mat;
	}

	bool BackgroundFilter::GlSetup()
	{
		Renderer::Init();

		m_Shader = BackInfi::Shader::Create("Background blend", kBasicVertex, kFragmentBackground);

		m_VertexBuffer = BackInfi::VertexArray::Create();

		float vertices[4 * 4] = {
			-1.0f, -1.0f,  /*bottom left*/   0.0f, 0.0f,  // bottom left
			1.0f,  -1.0f,  /*bottom right*/  1.0f, 0.0f,  // bottom right
			-1.0f, 1.0f,   /*top left*/      0.0f, 1.0f,  // top left
			1.0f,  1.0f,   /*top right*/     1.0f, 1.0f,  // top right
		};

		std::shared_ptr<BackInfi::VertexBuffer> vertexBuffer = BackInfi::VertexBuffer::Create(vertices, sizeof(vertices));
		BackInfi::BufferLayout layout = {
			{ BackInfi::ShaderDataType::Float2, "position" },
			{ BackInfi::ShaderDataType::Float2, "texture_coordinate" },
		};
		vertexBuffer->SetLayout(layout);
		m_VertexBuffer->AddVertexBuffer(vertexBuffer);

		uint32_t indices[6] = { 0, 1, 3, 3, 2, 0 };
		std::shared_ptr<BackInfi::IndexBuffer> indexBuffer = BackInfi::IndexBuffer::Create(indices, sizeof(indices) / sizeof(uint32_t));
		m_VertexBuffer->SetIndexBuffer(indexBuffer);

		BackInfi::FrameBufferSpecs specs;
		specs.Size          = m_Width * m_Height * 3;
		specs.Width         = m_Width;
		specs.Height        = m_Height;
		specs.Samples       = 4;
		specs.TextureFormat = TexFormat::RGB8;
		specs.TextureFilter = { TexFilterFormat::Linear, TexFilterFormat::LinearMipMapLinear };
		specs.TextureWrap   = { TexWrapFormat::ClampToEdge, TexWrapFormat::ClampToEdge };
		m_FrameBuffer       = BackInfi::FrameBuffer::Create(specs);

		m_Background        = cv::Mat(720, 1280, CV_8UC3, { 255, 0, 0 });

		m_InputTexture      = BackInfi::Texture2D::Create({ 4, 1280, 720, true, BackInfi::ImageFormat::RGB8 });
		m_MaskTexture       = BackInfi::Texture2D::Create({ 1, 256, 144, true, BackInfi::ImageFormat::R32 });
		m_BackgroundTexture = BackInfi::Texture2D::Create({ 4, 1280, 720, true, BackInfi::ImageFormat::RGB8 });

		m_Shader->Bind();
		m_Shader->SetInt("frame1", 1);
		m_Shader->SetInt("frame2", 2);
		m_Shader->SetInt("mask",   3);

		return true;
	}

	cv::Mat BackgroundFilter::BlendBackgroundAndForeground(const cv::Mat& inputData)
	{
		// opencv part --------------------------------------------------

		//cv::Mat mask_mat = m_Filter->backgroundMask;
		//if (mask_mat.channels() > 1)
		//{
		//   std::vector<cv::Mat> channels;
		//   cv::split(mask_mat, channels);
		//   mask_mat = channels[0];
		//}
		//
		//cv::Mat image;
		//cv::cvtColor(imageRGBA, image, cv::COLOR_BGRA2BGR);
		//cv::Mat output_mat(image.rows, image.cols, image.type());

		//const int invert_mask = false ? 1 : 0;
		//const int adjust_with_luminance = false ? 1 : 0;

		//cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range& range) {
		//   for (int i = range.start; i < range.end; i++) {
		//       for (int j = 0; j < image.cols; j++) {
		//           const float weight = mask_mat.at<float>(i, j) * (1.0 / 255.0);
		//           output_mat.at<cv::Vec3b>(i, j) =
		//               Blend(image.at<cv::Vec3b>(i, j), recolor, weight, invert_mask,
		//                   adjust_with_luminance);
		//       }
		//   }
		//   });

		//cv::flip(output_mat, output_mat, 0);
		//cv::cvtColor(output_mat, output_mat, cv::COLOR_BGR2RGB);

		// end of opencv part -------------------------------------------



		// testing opengl part ------------------------------------------

		m_FrameBuffer->Bind();
		RenderCommand::SetClearColor({ 0.1f, 0.1f, 0.1f, 1.0f });
		RenderCommand::Clear();

		Renderer::BeginScene();

		cv::flip(inputData, inputData, 0);

		m_MaskTexture->LoadTexture(m_BackgroundMask.data, m_BackgroundMask.step[0] * m_BackgroundMask.rows);
		m_InputTexture->LoadTexture(inputData.data, inputData.step[0] * inputData.rows);
		m_BackgroundTexture->LoadTexture(m_Background.data, m_Background.step[0] * m_Background.rows);

		m_InputTexture->Bind(1);
		m_BackgroundTexture->Bind(2);
		m_MaskTexture->Bind(3);
		Renderer::Submit(m_Shader, m_VertexBuffer);

		cv::Mat output_mat(
			m_FrameBuffer->GetSpecs().Height,
			m_FrameBuffer->GetSpecs().Width,
			CV_8UC3,
			m_FrameBuffer->ReadPixels(0, 0, 0)
		);
		cv::cvtColor(output_mat, output_mat, cv::COLOR_RGB2BGR);
		cv::flip(output_mat, output_mat, 0);

		Renderer::EndScene();

		m_FrameBuffer->UnBind();

		// end of testing opengl part -----------------------------------

		return output_mat;
	}

}