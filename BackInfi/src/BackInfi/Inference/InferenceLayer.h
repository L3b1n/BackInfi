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
#endif

#include <glm/glm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "BackInfi/Inference/InferenceConsts.h"
#include "BackInfi/Inference/Models/ModelRVM.h"
#include "BackInfi/Inference/Models/ModelSINET.h"
#include "BackInfi/Inference/Models/ModelSelfie.h"
#include "BackInfi/Inference/Models/ModelMediapipe.h"
#include "BackInfi/Inference/Models/ModelPPHumanSeg.h"

#include "BackInfi/Core/Layer.h"

namespace BackInfi
{

	// Settings structure
	struct InferenceSettings
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

	struct BackgroundRemoval1
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

		int MaskEveryXFrames           = 1;
		int MaskEveryXFramesCount      = 0;

		cv::Mat LastBackgroundMask;
		cv::Mat LastImage;
	};

	class InferenceLayer : public Layer
	{
	public:
		InferenceLayer();
		~InferenceLayer() = default;

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnUpdate(TimeStep ts) override;
		virtual void OnEvent(Event& event) override;

		void UpdateInputImage(const cv::Mat& input) { m_InputImage = input.clone(); }
		void UpdateSettings(const InferenceSettings& settings);

		cv::Mat GetBackgroundMask() { return m_BackgroundMask.clone(); }

	private:
		void UpdateFilter();

	private:
		cv::Mat                            m_InputImage;
		cv::Mat                            m_BackgroundMask;
		InferenceSettings                  m_Settings;

		std::unique_ptr<BackgroundRemoval1> m_Filter;
	};

}