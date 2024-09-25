#pragma once

#include "BackInfi.h"

#include "BackInfi/Inference/InferenceLayer.h"

#include <opencv2/opencv.hpp>

#include "BackInfi/Inference/InferenceConsts.h"
#include "BackInfi/Inference/Models/ModelRVM.h"
#include "BackInfi/Inference/Models/ModelSINET.h"
#include "BackInfi/Inference/Models/ModelSelfie.h"
#include "BackInfi/Inference/Models/ModelMediapipe.h"
#include "BackInfi/Inference/Models/ModelPPHumanSeg.h"

// Settings structure
struct InferenceSettings
{
	bool  EnableThreshold = false;
	bool  UseFloatMask = false;
	bool  EnableImageSimilarity = false;
	float ImageSimilarityThreshold = 0.0f;
	float Threshold = 0.0f;
	float ContourFilter = 0.0f;
	float SmoothContour = 0.0f;
	float Feather = 0.0f;
	float TemporalSmoothFactor = 0.0f;
	int   BlurBackground = 0;
	int   MaskEveryXFrames = 1;
	uint32_t NumThreads = 1;

	std::string UseGpu = USEGPU_CPU;
	std::string Model = MODEL_MEDIAPIPE;
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
	std::unique_ptr<BackInfi::Model> Model;

	// Filter part
	bool  IsDisabled = true;
	bool  UseFloatMask = false;
	bool  EnableThreshold = false;
	bool  EnableImageSimilarity = false;
	float ImageSimilarityThreshold = 0.0f;
	float Threshold = 0.0f;
	float ContourFilter = 0.0f;
	float SmoothContour = 0.0f;
	float Feather = 0.0f;
	float TemporalSmoothFactor = 0.0f;

	int MaskEveryXFrames = 1;
	int MaskEveryXFramesCount = 0;

	cv::Mat LastBackgroundMask;
	cv::Mat LastImage;
};

class BackInfiLayer : public BackInfi::Layer
{
public:
	BackInfiLayer(const BackInfi::InferenceSettings& settings);
	virtual ~BackInfiLayer() = default;

	virtual void OnAttach() override;
	virtual void OnDetach() override;

	virtual void OnUpdate(BackInfi::TimeStep ts) override;
	virtual void OnImGuiRender() override;
	virtual void OnEvent(BackInfi::Event& e) override;

private:
	void UpdateFilter();
	void OnUpdateInference(const cv::Mat& input);

private:
	int                                    m_Width;
	int                                    m_Height;
	bool                                   m_Blur;
	cv::Mat                                m_Background;
	cv::Mat                                m_PreviousMask;
	cv::Mat                                m_BackgroundMask;

	glm::vec2                              m_ViewportSize = { 0.0f, 0.0f };
	glm::vec2                              m_ViewportBounds[2];

	BackInfi::InferenceSettings            m_Settings;

	cv::VideoCapture                       m_Cap;

	std::shared_ptr<BackInfi::Shader>      m_Shader;

	std::shared_ptr<BackInfi::FrameBuffer> m_FrameBuffer;
	std::shared_ptr<BackInfi::VertexArray> m_VertexBuffer;

	std::shared_ptr<BackInfi::Texture>     m_MaskTexture;
	std::shared_ptr<BackInfi::Texture>     m_InputTexture;
	std::shared_ptr<BackInfi::Texture>     m_BackgroundTexture;

	std::unique_ptr<BackgroundRemoval1>    m_Filter;
};