#include "bcpch.h"

#include <glad/glad.h>
#include <opencv2/opencv.hpp>

#include "BackInfi/Core/Window.h"

#include "BackInfi/Inference/BackgroundFilter.h"

const int width = 1280;
const int height = 720;

int main()
{
	BackInfi::Logger::Init();
	
	std::unique_ptr<BackInfi::Window> Window = BackInfi::Window::Create(BackInfi::WindowProp("Test"));

	// To make sure the program does not quit running
	cv::VideoCapture cap;
	cap.open(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	cap.set(cv::CAP_PROP_FPS, 30);

	cv::namedWindow("Background remove app", cv::WINDOW_NORMAL);
	cv::resizeWindow("Background remove app", width, height);

	BackInfi::BackgroundFilter filter;
	filter.SetUp(height, width, false);

	BackInfi::Settings settings;
	settings.UseFloatMask             = true;
	settings.EnableThreshold          = true;
	settings.EnableImageSimilarity    = true;
	settings.Threshold                = 0.9f;
	settings.ImageSimilarityThreshold = 35.0f;
	settings.TemporalSmoothFactor     = 0.85f;
	settings.Feather                  = 0.025f;
	//m_settings.m_smoothContour			  = 0.25f;
	settings.Model                    = MODEL_MEDIAPIPE;
	settings.NumThreads               = 1;
	settings.BlurBackground           = false ? 6 : 0;

	filter.FilterUpdate(settings);
	filter.FilterActivate();
	//filter.LoadBackground("background.jpg");

	filter.GlSetup();

	int frames = 0;
	double fps = 0.0;
	float seconds = 0.0;
	bool load_flag = true;
	auto start_time = std::chrono::high_resolution_clock::now();

	while (load_flag)
	{
		cv::Mat temp;
		cap >> temp;

		filter.FilterVideoTick(temp);
		//filter.BlendSegmentationSmoothing(0.98);
		cv::Mat frame = filter.BlendBackgroundAndForeground(temp);

		//cv::Mat frame = filter.GetMask();

		Window->OnUpdate();

		frames++;
		auto end_time = std::chrono::high_resolution_clock::now();
		double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
		if (elapsed_time >= 1.0)
		{
			fps = frames / elapsed_time;
			frames = 0;
			start_time = std::chrono::high_resolution_clock::now();
		}

		std::stringstream ss;
		ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
		cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		cv::imshow("Background remove app", frame);

		const int pressed_key = cv::waitKey(5);
		if (pressed_key == 27) load_flag = false;
		if (cv::getWindowProperty("Background remove app", cv::WND_PROP_VISIBLE) < 1) load_flag = false;
	}
	cap.release();
	cv::destroyAllWindows();
	std::cin.get();
	return 0;
}