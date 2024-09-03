#pragma once
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

#ifdef BC_PLATFORM_APPLE
#include <coreml_provider_factory.h>
#endif

#ifdef BC_PLATFORM_LINUX
#include <tensorrt_provider_factory.h>
#endif

#ifdef BC_PLATFORM_WINDOWS
//#include <dml_provider_factory.h>
#include <wchar.h>
#endif

#include <glad/glad.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "BackInfi/Inference/GlPrograms.h"
#include "BackInfi/Inference/Models/ModelRVM.h"
#include "BackInfi/Inference/Models/ModelSINET.h"
#include "BackInfi/Inference/Models/ModelSelfie.h"
#include "BackInfi/Inference/Models/ModelMediapipe.h"
#include "BackInfi/Inference/Models/ModelPPHumanSeg.h"

#include "BackInfi/Renderer/Shader.h"
#include "BackInfi/Renderer/Buffer.h"
#include "BackInfi/Renderer/Texture.h"
#include "BackInfi/Renderer/Renderer.h"
#include "BackInfi/Renderer/FrameBuffer.h"
#include "BackInfi/Renderer/VertexArray.h"

namespace BackInfi
{

	// Settings structure
	struct Settings
	{
		bool  EnableThreshold          = false;
		bool  UseFloatMask             = false;
		bool  EnableImageSimilarity    = false;
		float ImageSimilarityThreshold = 0.0f;
		float Threshold                = 0.0f;
		float ContourFilter            = 0.0f;
		float SmoothContour            = 0.0f;
		float Feather                  = 0.0f;
		float TemporalSmoothFactor     = 0.0f;
		int   BlurBackground           = 0;
		int   MaskEveryXFrames         = 1;
		uint32_t NumThreads            = 1;

		std::string UseGpu             = USEGPU_CPU;
		std::string Model              = MODEL_MEDIAPIPE;
	};

	struct BackgroundRemoval
	{
		// ORTModel data
		std::unique_ptr<Ort::Session>        OnnxSession;
		std::unique_ptr<Ort::Env>            OnnxEnv;
		std::vector<Ort::AllocatedStringPtr> InputNames;
		std::vector<Ort::AllocatedStringPtr> OutputNames;
		std::vector<Ort::Value>              InputTensor;
		std::vector<Ort::Value>              OutputTensor;
		std::vector<std::vector<int64_t>>    InputDims;
		std::vector<std::vector<int64_t>>    OutputDims;
		std::vector<std::vector<float>>      OutputTensorValues;
		std::vector<std::vector<float>>      InputTensorValues;

		// Model part
		uint32_t               NumThreads;
		std::string            UseGPU;
		std::string            ModelSelection;
		std::unique_ptr<Model> Model;

		// Filter part
		bool  IsDisabled               = true;
		bool  UseFloatMask             = false;
		bool  EnableThreshold          = false;
		bool  EnableImageSimilarity    = false;
		float ImageSimilarityThreshold = 0.0f;
		float Threshold                = 0.0f;
		float ContourFilter            = 0.0f;
		float SmoothContour            = 0.0f;
		float Feather                  = 0.0f;
		float TemporalSmoothFactor     = 0.0f;

		int BlurBackground             = 0;
		int MaskEveryXFrames           = 0;
		int MaskEveryXFramesCount      = 0;

		cv::Mat LastBackgroundMask;
		cv::Mat LastImage;
	};

	class BackgroundFilter
	{
	public:	
		BackgroundFilter();
		~BackgroundFilter();

		void SetUp(const int&  height, const int&  width, const bool& blur);

		inline void FilterActivate() { m_Filter->IsDisabled = false; }

		inline void FilterDeactivate() { m_Filter->IsDisabled = true; }

		void LoadBackground(const std::string path);

		void FilterUpdate(const Settings& settings);

		void FilterVideoTick(const cv::Mat& imageBGR);

		void BlendSegmentationSmoothing(const double combine_with_previous_ratio_);

		cv::Mat BlendBackgroundAndForeground(const cv::Mat& inputData);

		inline cv::Mat GetMask() { return m_BackgroundMask; }

		bool GlSetup();

	private:
		int                                    m_Width;
		int                                    m_Height;
		bool                                   m_Blur;
		cv::Mat                                m_Background;
		cv::Mat                                m_PreviousMask;
		cv::Mat                                m_BackgroundMask;

		std::unique_ptr<BackgroundRemoval>     m_Filter;

		std::shared_ptr<BackInfi::Shader>      m_Shader;

		std::shared_ptr<BackInfi::FrameBuffer> m_FrameBuffer;
		std::shared_ptr<BackInfi::VertexArray> m_VertexBuffer;

		std::shared_ptr<BackInfi::Texture>     m_MaskTexture;
		std::shared_ptr<BackInfi::Texture>     m_InputTexture;
		std::shared_ptr<BackInfi::Texture>     m_BackgroundTexture;

		inline cv::Vec3b Blend(const cv::Vec3b& color1, const cv::Vec3b& color2,
			float weight, int invert_mask,
			int adjust_with_luminance)
		{
			weight = (1 - invert_mask) * weight + invert_mask * (1.0f - weight);

			float luminance =
				(1 - adjust_with_luminance) * 1.0f +
				adjust_with_luminance *
				(color1[0] * 0.299 + color1[1] * 0.587 + color1[2] * 0.114) / 255;

			float mix_value = weight * luminance;

			return color1 * (1.0 - mix_value) + color2 * mix_value;
		}
	};

}
