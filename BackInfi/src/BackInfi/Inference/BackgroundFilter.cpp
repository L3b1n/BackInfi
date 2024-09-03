#include "bcpch.h"
#include "BackgroundFilter.h"
#include "ort-utils/ort-session-utils.h"
#include <opencv2/opencv.hpp>

namespace BackInfi
{

	void BackgroundFilter::filterUpdate(const bool enableThreshold_, const float threshold_,
		const float contourFilter_, const float smoothContour_,
		const float feather_, const int maskEveryXframes_,
		const int64_t blurBackground_, const std::string useGpu_,
		const std::string model_, const uint32_t numThreads_)
	{

		tf->enableThreshold = enableThreshold_;
		tf->threshold = 0.9f /*float [0, 1]*/;

		tf->contourFilter = 0.05 /*float [0, 1]*/;
		tf->smoothContour = 0.6 /*float [0, 1]*/;
		tf->feather = 0.0 /*float [0, 1]*/;
		tf->maskEveryXFrames = 1;
		tf->maskEveryXFramesCount = (int)(0);
		tf->blurBackground = 0;

		const std::string newUseGpu = useGpu_ /*USEGPU_CPU for macos and linux*/;
		const std::string newModel = MODEL_MEDIAPIPE /*MODEL_***/;
		const uint32_t newNumThreads = 1 /*uint32_t max 3, by defolt 1*/;

		if (tf->modelSelection.empty() || tf->modelSelection != newModel ||
			tf->useGPU != newUseGpu || tf->numThreads != newNumThreads)
		{
			// Re-initialize model if it's not already the selected one or switching inference device
			tf->modelSelection = newModel;
			tf->useGPU = newUseGpu;
			tf->numThreads = newNumThreads;

			if (tf->modelSelection == MODEL_SINET) {
				tf->model.reset(new ModelSINET);
			}
			if (tf->modelSelection == MODEL_SELFIE) {
				tf->model.reset(new ModelSelfie);
			}
			if (tf->modelSelection == MODEL_MEDIAPIPE) {
				tf->model.reset(new ModelMediapipe);
			}
			if (tf->modelSelection == MODEL_RVM) {
				tf->model.reset(new ModelRVM);
			}
			if (tf->modelSelection == MODEL_PPHUMANSEG) {
				tf->model.reset(new ModelPPHumanSeg);
			}

			createOrtSession(tf);
		}
	}

	void BackgroundFilter::processImageForBackground(
		const cv::Mat& imageBGRA,
		cv::Mat& backgroundMask)
	{
		cv::Mat outputImage;
		if (!runFilterModelInference(tf, imageBGRA, outputImage)) {
			return;
		}
		// Assume outputImage is now a single channel, uint8 image with values between 0 and 255

		// If we have a threshold, apply it. Otherwise, just use the output image as the mask
		if (tf->enableThreshold) {
			// We need to make tf->threshold (float [0,1]) be in that range
			const uint8_t threshold_value = (uint8_t)(tf->threshold * 255.0f);
			backgroundMask = outputImage < threshold_value;
		}
		else {
			backgroundMask = 255 - outputImage;
		}
	}

	void BackgroundFilter::filterVideoTick(
		const int height, const int width,
		const int type, unsigned char* data)
	{

		if (tf->isDisabled) { std::cerr << "Error! Rendering of background filter is disabled!\n"; return; }

		cv::Mat input_mat(height, width, type, data), input_matRGBA;
		cv::cvtColor(input_mat, input_matRGBA, cv::COLOR_BGR2RGBA);
		tf->inputRGBA = input_matRGBA;

		if (tf->inputRGBA.empty()) { std::cerr << "Error! Input image is empty!\n"; return; }

		cv::Mat imageRGBA;
		{
			std::unique_lock<std::mutex> lock(tf->inputRGBALock, std::try_to_lock);
			if (!lock.owns_lock()) {
				return;
			}
			imageRGBA = tf->inputRGBA.clone();
		}
		//cv::imshow("InputRGBA", imageRGBA);

		if (tf->backgroundMask.empty()) {
			// First frame. Initialize the background mask.
			tf->backgroundMask = cv::Mat(imageRGBA.size(), CV_8UC1, cv::Scalar(255));
		}

		tf->maskEveryXFramesCount++;
		tf->maskEveryXFramesCount %= tf->maskEveryXFrames;

		try {
			if (tf->maskEveryXFramesCount != 0 && !tf->backgroundMask.empty()) {
				// We are skipping processing of the mask for this frame.
				// Get the background mask previously generated.
				; // Do nothing
			}
			else {
				cv::Mat backgroundMask;

				// Process the image to find the mask.
				processImageForBackground(imageRGBA, backgroundMask);

				//cv::resize(backgroundMask, backgroundMask, cv::Size(width / 8, height / 8));

				// Contour processing
				// Only applicable if we are thresholding (and get a binary image)
				if (tf->enableThreshold) {
					if (!tf->contourFilter > 0.0 && !tf->contourFilter < 1.0) {
						std::vector<std::vector<cv::Point>> contours;
						findContours(backgroundMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
						std::vector<std::vector<cv::Point>> filteredContours;
						const int64_t contourSizeThreshold =
							(int64_t)(backgroundMask.total() * tf->contourFilter);
						for (auto& contour : contours) {
							if (cv::contourArea(contour) > contourSizeThreshold) {
								filteredContours.push_back(contour);
							}
						}
						backgroundMask.setTo(0);
						cv::drawContours(backgroundMask, filteredContours, -1, cv::Scalar(255), -1);
					}

					if (!tf->smoothContour > 0.0) {
						int k_size = (int)(3 + 11 * tf->smoothContour);
						k_size += k_size % 2 == 0 ? 1 : 0;
						cv::stackBlur(backgroundMask, backgroundMask, cv::Size(k_size, k_size));
					}

					//cv::resize(backgroundMask, backgroundMask, imageRGBA.size());

					cv::blur(backgroundMask, backgroundMask, cv::Size(7, 7));
					// Additional contour processing at full resolution
					if (!tf->smoothContour > 0.0) {
						// If the mask was smoothed, apply a threshold to get a binary mask
						backgroundMask = backgroundMask > 128;
					}

					if (!tf->feather > 0.0) {
						// Feather (blur) the mask
						int k_size = (int)(40 * tf->feather);
						k_size += k_size % 2 == 0 ? 1 : 0;
						cv::dilate(backgroundMask, backgroundMask, cv::Mat(), cv::Point(-1, -1), k_size / 3);
						cv::boxFilter(backgroundMask, backgroundMask, tf->backgroundMask.depth(),
							cv::Size(k_size, k_size));
					}
				}

				cv::flip(backgroundMask, backgroundMask, 0); // This is not for LVC realization

				//backgroundMask.convertTo(backgroundMask, CV_32F); // This is need for opencv realisation, otherwise delete this line
				backgroundMask.convertTo(backgroundMask, CV_32F); // This is need for opengl realisation, otherwise delete this line

				// Resize the size of the mask back to the size of the original input.
				cv::resize(backgroundMask, backgroundMask, imageRGBA.size());

				//cv::Mat temp;
				//double sigma_space_ = 1;
				//double sigma_color_ = 0.1;
				//if (backgroundMask.channels() == 1 || backgroundMask.channels() == 3) {
				//    cv::bilateralFilter(backgroundMask, temp, /*d=*/sigma_space_ * 2.0,
				//        sigma_color_, sigma_space_);
				//}

				// Save the mask for the next frame
				backgroundMask.copyTo(tf->backgroundMask);
			}
		}
		catch (const Ort::Exception& e) {
			std::cerr << "Error! " << e.what() << "\n";
			// TODO: Fall back to CPU if it makes sense
		}
		catch (const std::exception& e) {
			std::cerr << "Error! " << e.what() << "\n";
		}
	}

	void BackgroundFilter::blendSegmentationSmoothing(
		const double combine_with_previous_ratio_)
	{
		if (tf->backgroundMask.empty()) { std::cerr << "Error! Background mask is empty!\n"; }

		current_mask = tf->backgroundMask;
		if (previous_mask.empty()) { previous_mask = tf->backgroundMask; }

		if (previous_mask.type() != current_mask.type())
		{
			std::cerr << "Warning: mixing input format types: " << previous_mask.type()
				<< " != " << current_mask.type() << "\n";
			return;
		}

		if (previous_mask.rows != current_mask.rows) { return; }
		if (previous_mask.cols != current_mask.cols) { return; }

		// Setup destination image.
		cv::Mat output_mat(current_mask.rows, current_mask.cols, current_mask.type());
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
				const float t = new_mask_value - 0.5f;
				const float x = t * t;

				const float uncertainty =
					1.0f -
					std::min(1.0f, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))));

				return new_mask_value + (prev_mask_value - new_mask_value) *
					(uncertainty * combine_with_previous_ratio_);
		};

		// Write directly to the first channel of output.
		for (int i = 0; i < output_mat.rows; ++i) {
			float* out_ptr = output_mat.ptr<float>(i);
			const float* curr_ptr = current_mask.ptr<float>(i);
			const float* prev_ptr = previous_mask.ptr<float>(i);
			for (int j = 0; j < output_mat.cols; ++j) {
				const float new_mask_value = curr_ptr[j];
				const float prev_mask_value = prev_ptr[j];
				out_ptr[j] = blending_fn(prev_mask_value, new_mask_value);
			}
		}

		tf->backgroundMask = output_mat;
		previous_mask = output_mat;
	}

	cv::Mat BackgroundFilter::TextureToMat(GLuint texture_id)
	{
		glBindTexture(GL_TEXTURE_2D, texture_id);
		unsigned char* texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 4);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_bytes);

		return cv::Mat(height, width, CV_8UC4, texture_bytes); // CV_8UC3 for simple image, ..C4 for mask
	}

	cv::Mat BackgroundFilter::BufferToMat()
	{
		cv::Mat output_mat(tf->inputRGBA.size(), CV_8UC4); // because of RGBA
		// use fast 4-byte alignment (default anyway) if possible
		glPixelStorei(GL_PACK_ALIGNMENT, (output_mat.step & 3) ? 1 : 4);

		// set length of one complete row in destination data (doesn't need to equal img.cols)
		glPixelStorei(GL_PACK_ROW_LENGTH, (GLint)(output_mat.step / output_mat.elemSize()));
		glReadPixels(0, 0, output_mat.cols, output_mat.rows, GL_RGBA, GL_UNSIGNED_BYTE, output_mat.data);
		//cv::flip(output_mat, output_mat, 0);
		cv::cvtColor(output_mat, output_mat, cv::COLOR_RGBA2BGR);

		return output_mat;
	}

	bool BackgroundFilter::GlSetup(const int mask_channel)
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
		specs.Size          = width * height * 3;
		specs.Width         = width;
		specs.Height        = height;
		specs.Samples       = 4;
		specs.TextureFormat = TexFormat::RGB8;
		specs.TextureFilter = { TexFilterFormat::Linear, TexFilterFormat::LinearMipMapLinear };
		specs.TextureWrap   = { TexWrapFormat::ClampToEdge, TexWrapFormat::ClampToEdge };
		m_FrameBuffer       = BackInfi::FrameBuffer::Create(specs);

		m_InputTexture      = BackInfi::Texture2D::Create({ 4, 1280, 720, true, BackInfi::ImageFormat::RGBA8 });
		m_MaskTexture       = BackInfi::Texture2D::Create({ 1, 1280, 720, true, BackInfi::ImageFormat::R8 });
		m_BackgroundTexture = BackInfi::Texture2D::Create({ 4, 1280, 720, true, BackInfi::ImageFormat::RGBA8 });

		m_Shader->Bind();
		m_Shader->SetInt("frame1", 1);
		m_Shader->SetInt("frame2", 2);
		m_Shader->SetInt("mask",   3);

		return true;
	}

	cv::Mat BackgroundFilter::blendBackgroundAndForeground()
	{

		if (tf->inputRGBA.empty()) { BC_CORE_ERROR("Error! Input image is empty!"); }

		cv::Mat imageRGBA;
		{
			std::unique_lock<std::mutex> lock(tf->inputRGBALock, std::try_to_lock);
			if (!lock.owns_lock()) {
				return tf->inputRGBA;
			}
			imageRGBA = tf->inputRGBA.clone();
		}
		cv::flip(imageRGBA, imageRGBA, 0);
		cv::blur(imageRGBA, background, cv::Size(19, 19));

		cv::Mat mask_mat = tf->backgroundMask;

		// opencv part --------------------------------------------------

		//cv::Mat mask_mat = tf->backgroundMask;
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

		m_MaskTexture->LoadTexture(mask_mat.data, mask_mat.step[0] * mask_mat.rows);
		m_InputTexture->LoadTexture(imageRGBA.data, imageRGBA.step[0] * imageRGBA.rows);
		m_BackgroundTexture->LoadTexture(background.data, background.step[0] * background.rows);

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