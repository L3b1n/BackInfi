#pragma once
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#endif // _WIN32

#include <glad/glad.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <iostream>
#include <filesystem>

#include "BackInfi/Renderer/Shader.h"

#include "models/ModelSINET.h"
#include "models/ModelMediapipe.h"
#include "models/ModelSelfie.h"
#include "models/ModelRVM.h"
#include "models/ModelPPHumanSeg.h"
#include "FilterData.h"
#include "consts.h"

// A square covering the full clip space.
static const GLfloat kBasicSquareVertices[] = {
	-1.0f, -1.0f,  // bottom left
	1.0f,  -1.0f,  // bottom right
	-1.0f, 1.0f,   // top left
	1.0f,  1.0f,   // top right
};

// Places a texture on kBasicSquareVertices with normal alignment.
static const GLfloat kBasicTextureVertices[] = {
	0.0f, 0.0f,  // bottom left
	1.0f, 0.0f,  // bottom right
	0.0f, 1.0f,  // top left
	1.0f, 1.0f,  // top right
};

class BackgroundFilter
{
public:	
	BackgroundFilter() {
		tf = new BackgroundRemovalFilter();
		std::string instanceName{"background-removal-inference"};
		tf->env.reset(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str()));
		tf->modelSelection = MODEL_MEDIAPIPE;

		shader = BackInfi::Shader::Create("Background blend", kBasicVertexShader, frag_src_tex);
	}

	BackgroundFilter(
		const int height_,
		const int width_) 
			: width(width_),
			height(height_)
	{
		tf = new BackgroundRemovalFilter();
		std::string instanceName{"background-removal-inference"};
		tf->env.reset(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str()));
		tf->modelSelection = MODEL_MEDIAPIPE;

		shader = BackInfi::Shader::Create("Background blend", kBasicVertexShader, frag_src_tex);
	}

	void filterActivate() { tf->isDisabled = false; }

	void filterDeactivate() { tf->isDisabled = true; }

	void loadBackground(const std::string path) {
		std::string backgroundImagePath = std::filesystem::current_path().append(path).string();
		background = cv::imread(backgroundImagePath);
		cv::resize(background, background, cv::Size(width, height));
		cv::cvtColor(background, background, cv::COLOR_BGR2RGBA);
		cv::flip(background, background, 0);
	}

    void filterUpdate(const bool enableThreshold_, const float threshold_ = 0.025,
        const float contourFilter_ = 0.025, const float smoothContour_ = 0.05,
        const float feather_ = 0.05, const int maskEveryXframes_ = 1,
        const int64_t blurBackground_ = 0, const std::string useGpu_ = USEGPU_CPU,
        const std::string model_ = MODEL_MEDIAPIPE, const uint32_t numThreads_ = 1);

	void filterVideoTick(const int height, const int width,
		const int type, unsigned char* data);

	void blendSegmentationSmoothing(const double combine_with_previous_ratio_);

	//void blendBackgroundAndForeground(unsigned char* data);
	cv::Mat blendBackgroundAndForeground();

	bool GlSetup(const int mask_channel);

	GLuint program_ = 0;
	GLuint VAO;

private:
	int width = 0;
	int height = 0;
	cv::Mat background;
	cv::Mat current_mask;
	cv::Mat previous_mask;
	const cv::Vec3b recolor = { 255, 0, 0 };

    BackgroundRemovalFilter *tf;

    //GLuint program_ = 0;

	std::shared_ptr<BackInfi::Shader> shader;
    GLint unif_frame1_;
    GLint unif_frame2_;
    GLint unif_mask_;
	
    bool GlRender(const float mask_const);

	cv::Mat BufferToMat();
    cv::Mat TextureToMat(GLuint texture_id);
    void BindCVMatToGLTexture(cv::Mat& image, GLuint& imageTexture);

	void processImageForBackground(const cv::Mat& imageBGRA, cv::Mat& backgroundMask);

	inline cv::Vec3b Blend(const cv::Vec3b& color1, const cv::Vec3b& color2,
		float weight, int invert_mask,
		int adjust_with_luminance) {
		weight = (1 - invert_mask) * weight + invert_mask * (1.0f - weight);

		float luminance =
			(1 - adjust_with_luminance) * 1.0f +
			adjust_with_luminance *
			(color1[0] * 0.299 + color1[1] * 0.587 + color1[2] * 0.114) / 255;

		float mix_value = weight * luminance;

		return color1 * (1.0 - mix_value) + color2 * mix_value;
	}
};