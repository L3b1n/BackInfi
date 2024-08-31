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
	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	cap.set(cv::CAP_PROP_FPS, 30);

	cv::namedWindow("Background remove app", cv::WINDOW_NORMAL);
	cv::resizeWindow("Background remove app", width, height);

	BackInfi::BackgroundFilter filter(height, width);
	filter.filterUpdate(true);
	filter.filterActivate();
	//filter.loadBackground("background.jpg");

	const auto mask_channel = 1;
	filter.GlSetup(mask_channel);

	glDisable(GL_BLEND);

	int frames = 0;
	double fps = 0.0;
	float seconds = 0.0;
	bool load_flag = true;
	auto start_time = std::chrono::high_resolution_clock::now();

	while (load_flag)
	{
		cv::Mat temp;
		cap >> temp;

		// processInput(window);

		filter.filterVideoTick(temp.rows, temp.cols, temp.type(), temp.data);
		filter.blendSegmentationSmoothing(0.98);
		cv::Mat frame = filter.blendBackgroundAndForeground();
        
		//unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * 1920 * 1080 * 4);
		//filter.blendBackgroundAndForeground(data);
		//cv::Mat frame(temp.size(), temp.type(), data);
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