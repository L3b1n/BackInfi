#include "bcpch.h"

#include "BackInfiLayer.h"

//#include "BackInfi/Inference/InferenceLayer.h"

#include "BackInfi/Inference/Models/ModelsWeights/MediapipeWeights.h"

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

BackInfiLayer::BackInfiLayer(const BackInfi::InferenceSettings& settings)
	: Layer("BackInfiLayer"), m_Settings(settings)
{
	m_Width  = 1280;
	m_Height = 720;

	m_Cap.open(0);
	m_Cap.set(cv::CAP_PROP_FRAME_WIDTH, m_Width);
	m_Cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_Height);
	m_Cap.set(cv::CAP_PROP_FPS, 30);

	const std::string kBasicVertex = R"(
		#version 330

		layout(location = 0) in vec4 position;
		layout(location = 1) in vec4 texture_coordinate;

		// texture coordinate for fragment shader (will be interpolated)
		out vec2 sample_coordinate;

		void main() {
			gl_Position = position;
			sample_coordinate = texture_coordinate.xy;
		}
	)";

	const std::string kFragmentBackground = R"(
		#version 330

		in vec2 sample_coordinate;
		out vec3 frag_out;

		uniform sampler2D frame1;
		uniform sampler2D frame2;
		uniform sampler2D mask;

		void main()
		{
			vec4 inputRGBA = texture(frame1, sample_coordinate);
			inputRGBA.bgr = max(vec3(0.0, 0.0, 0.0), inputRGBA.rgb / inputRGBA.a);

			vec2 texel_size = 1.0 / textureSize(mask, 0);
			float maskValue = texture(mask, sample_coordinate).r;
			vec3 maskTexture = vec3(maskValue);

			float a = (1.0 - maskTexture.r) * inputRGBA.a;
			// Because of output type I want to get back
			frag_out = inputRGBA.rgb * a + texture(frame2, sample_coordinate).bgr * (1.0 - a);
		}
	)";
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
	specs.TextureFormat = BackInfi::TexFormat::RGB8;
	specs.TextureFilter = { BackInfi::TexFilterFormat::Linear, BackInfi::TexFilterFormat::LinearMipMapLinear };
	specs.TextureWrap   = { BackInfi::TexWrapFormat::ClampToEdge, BackInfi::TexWrapFormat::ClampToEdge };
	m_FrameBuffer       = BackInfi::FrameBuffer::Create(specs);

	m_InputTexture      = BackInfi::Texture2D::Create({ 4, 1280, 720, true, BackInfi::ImageFormat::RGB8 });
	m_MaskTexture       = BackInfi::Texture2D::Create({ 1, 256, 144, true, BackInfi::ImageFormat::R32 });
	m_BackgroundTexture = BackInfi::Texture2D::Create({ 4, 1280, 720, true, BackInfi::ImageFormat::RGB8 });

	m_BackgroundTexture->LoadTexture(cv::Mat(720, 1280, CV_8UC3, { 255, 0, 0 }).data, 720 * 1280 * 3);

	m_Shader->Bind();
	m_Shader->SetInt("frame1", 1);
	m_Shader->SetInt("frame2", 2);
	m_Shader->SetInt("mask", 3);
}

void BackInfiLayer::OnAttach()
{
	m_Filter = std::make_unique<BackgroundRemoval1>();
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

void BackInfiLayer::OnDetach()
{
	BC_CORE_TRACE("Excit inference layer!");
	m_Cap.release();
}

void BackInfiLayer::OnUpdate(BackInfi::TimeStep ts)
{
	cv::Mat frame;
	m_Cap >> frame;

	OnUpdateInference(frame);

	//m_FrameBuffer->Bind();
	BackInfi::RenderCommand::SetClearColor({ 0.1f, 0.1f, 0.1f, 1.0f });
	BackInfi::RenderCommand::Clear();

	BackInfi::Renderer::BeginScene();

	cv::flip(frame, frame, 0);

	m_MaskTexture->LoadTexture(m_BackgroundMask.data, m_BackgroundMask.step[0] * m_BackgroundMask.rows);
	m_InputTexture->LoadTexture(frame.data, frame.step[0] * frame.rows);

	m_InputTexture->Bind(1);
	m_BackgroundTexture->Bind(2);
	m_MaskTexture->Bind(3);
	BackInfi::Renderer::Submit(m_Shader, m_VertexBuffer);

	//cv::Mat output_mat(
	//	m_FrameBuffer->GetSpecs().Height,
	//	m_FrameBuffer->GetSpecs().Width,
	//	CV_8UC3,
	//	m_FrameBuffer->ReadPixels(0, 0, 0)
	//);
	//cv::cvtColor(output_mat, output_mat, cv::COLOR_RGB2BGR);
	//cv::flip(output_mat, output_mat, 0);

	BackInfi::Renderer::EndScene();

	//m_FrameBuffer->UnBind();
}

void BackInfiLayer::OnUpdateInference(const cv::Mat& input)
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
	if (input.data == nullptr)
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
		input,
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
				m_Filter->UseFloatMask ? BackInfi::Model::Activation::SOFTMAX : BackInfi::Model::Activation::NONE,
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

void BackInfiLayer::UpdateFilter()
{
	m_Filter->EnableThreshold = m_Settings.EnableThreshold;
	m_Filter->Threshold = m_Settings.Threshold;

	m_Filter->EnableImageSimilarity = m_Settings.EnableImageSimilarity;
	m_Filter->ImageSimilarityThreshold = m_Settings.ImageSimilarityThreshold;

	m_Filter->UseFloatMask = m_Settings.UseFloatMask;

	m_Filter->ContourFilter = m_Settings.ContourFilter;
	m_Filter->SmoothContour = m_Settings.SmoothContour;
	m_Filter->Feather = m_Settings.Feather;
	m_Filter->TemporalSmoothFactor = m_Settings.TemporalSmoothFactor;

	m_Filter->MaskEveryXFrames = m_Settings.MaskEveryXFrames;
	m_Filter->MaskEveryXFramesCount = (int)(0);

	if (m_Filter->ModelSelection.empty() || m_Filter->ModelSelection != m_Settings.Model ||
		m_Filter->UseGPU != m_Settings.UseGpu || m_Filter->NumThreads != m_Settings.NumThreads)
	{
		// Re-initialize Model if it's not already the selected one or switching inference device
		m_Filter->ModelSelection = m_Settings.Model;
		m_Filter->UseGPU = m_Settings.UseGpu;
		m_Filter->NumThreads = m_Settings.NumThreads;

		if (m_Filter->ModelSelection == MODEL_SINET)
		{
			m_Filter->Model.reset(new BackInfi::ModelSINET);
		}
		if (m_Filter->ModelSelection == MODEL_SELFIE)
		{
			m_Filter->Model.reset(new BackInfi::ModelSelfie);
		}
		if (m_Filter->ModelSelection == MODEL_MEDIAPIPE)
		{
			m_Filter->Model.reset(new BackInfi::ModelMediapipe);
		}
		if (m_Filter->ModelSelection == MODEL_RVM) {
			m_Filter->Model.reset(new BackInfi::ModelRVM);
		}
		if (m_Filter->ModelSelection == MODEL_PPHUMANSEG)
		{
			m_Filter->Model.reset(new BackInfi::ModelPPHumanSeg);
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

void BackInfiLayer::OnImGuiRender()
{
}

void BackInfiLayer::OnEvent(BackInfi::Event& e)
{
}
