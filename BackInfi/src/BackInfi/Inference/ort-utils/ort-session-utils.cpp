#include "bcpch.h"
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

//#ifdef __linux__
//#include <tensorrt_provider_factory.h>
//#endif

#ifdef _WIN32
//#include <dml_provider_factory.h>
#include <wchar.h>
#include <iostream>
#endif // _WIN32

#include "BackInfi/Inference/ort-utils/ort-session-utils.h"
#include "BackInfi/Inference/GlPrograms.h"
#include "BackInfi/Inference/Models/ModelsWeights/MediapipeWeights.h"
//#include "BackInfi/Inference/Models/ModelsWeights/selfie_segmentation.h"
//#include "BackInfi/Inference/Models/ModelsWeights/SINet_Softmax_simple.h"
//#include "BackInfi/Inference/Models/ModelsWeights/rvm_mobilenetv3_fp32.h"
//#include "BackInfi/Inference/Models/ModelsWeights/pphumanseg_fp32.h"

namespace BackInfi
{

	void createOrtSession(BackgroundRemovalFilter* tf)
	{
		if (tf->model.get() == nullptr)
		{
			BC_CORE_ERROR("Error! Model object is not initialized!");
			return;
		}

		Ort::SessionOptions sessionOptions;

		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		if (tf->useGPU != USEGPU_CPU)
		{
			sessionOptions.DisableMemPattern();
			sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
		}
		else
		{
			sessionOptions.SetInterOpNumThreads(tf->numThreads);
			sessionOptions.SetIntraOpNumThreads(tf->numThreads);
		}

		char* modelSelection_rawPtr = (char*)tf->modelSelection.c_str();

		if (modelSelection_rawPtr == nullptr)
		{
			BC_CORE_ERROR("Error! Unable to get model filename {0} from plugin!", tf->modelSelection.c_str());
			return;
		}

		//if (tf->modelSelection == MODEL_SINET) {
		//    tf->modelInfo = SINet_Softmax_simple_onnx;
		//    tf->modelSize = SINet_Softmax_simple_onnx_len;
		//}
		//if (tf->modelSelection == MODEL_SELFIE) {
		//    tf->modelInfo = selfie_segmentation_onnx;
		//    tf->modelSize = selfie_segmentation_onnx_len;
		//}
		if (tf->modelSelection == MODEL_MEDIAPIPE)
		{
			tf->modelInfo = mediapipe_onnx;
			tf->modelSize = mediapipe_onnx_len;
		}
		//if (tf->modelSelection == MODEL_RVM) {
		//    tf->modelInfo = rvm_mobilenetv3_fp32_onnx;
		//    tf->modelSize = rvm_mobilenetv3_fp32_onnx_len;
		//}
		//if (tf->modelSelection == MODEL_PPHUMANSEG) {
		//    tf->modelInfo = pphumanseg_fp32_onnx;
		//    tf->modelSize = pphumanseg_fp32_onnx_len;
		//}

		try {
#ifdef __linux__
			if (tf->useGPU == USEGPU_TENSORRT)
			{
				Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
			}
#endif
#ifdef _WIN32
			if (tf->useGPU == USEGPU_DML)
			{
				/*auto &api = Ort::GetApi();
				OrtDmlApi *dmlApi = nullptr;
				Ort::ThrowOnError(
				  api.GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dmlApi));
				Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));*/
				//OrtOpenVINOProviderOptions options;
				//options.device_type = "CPU_FP32"; //Other options are: GPU_FP32, GPU_FP16, MYRIAD_FP16
				//std::cout << "OpenVINO device type is set to: " << options.device_type << std::endl;
				//sessionOptions.AppendExecutionProvider_OpenVINO(options);
			}
#endif
#if defined(__APPLE__)
			if (tf->useGPU == USEGPU_COREML)
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
			//tf->session.reset(new Ort::Session(*tf->env, modelFilepath_ws.c_str(), sessionOptions));
			// 
			tf->session.reset(new Ort::Session(*tf->env, tf->modelInfo, tf->modelSize, sessionOptions));
		}
		catch (const std::exception& e)
		{
			BC_CORE_ERROR("Error! {0}", e.what());
			return;
		}

		Ort::AllocatorWithDefaultOptions allocator;

		tf->model->PopulateInputOutputNames(tf->session, tf->inputNames, tf->outputNames);

		if (!tf->model->PopulateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims))
		{
			BC_CORE_ERROR("Error! Unable to get model input and output shapes!");
			return;
		}

		// Allocate buffers
		tf->model->AllocateTensorBuffers(
			tf->inputDims,
			tf->outputDims,
			tf->outputTensorValues,
			tf->inputTensorValues,
			tf->inputTensor,
			tf->outputTensor
		);
	}

	bool runFilterModelInference(BackgroundRemovalFilter* tf, const cv::Mat& imageRGBA, cv::Mat& output)
	{
		if (tf->session.get() == nullptr)
		{
			BC_CORE_ERROR("Error! Session isn't initialized!");
			return false;
		}
		if (tf->model.get() == nullptr)
		{
			BC_CORE_ERROR("Error! Model isn't initialized!");
			return false;
		}

		// To RGB
		cv::Mat imageRGB;
		cv::cvtColor(imageRGBA, imageRGB, cv::COLOR_RGBA2RGB);

		// Resize to network input size
		uint32_t inputWidth, inputHeight;
		tf->model->GetNetworkInputSize(tf->inputDims, inputWidth, inputHeight);

		cv::Mat resizedImageRGB;
		cv::resize(imageRGB, resizedImageRGB, cv::Size(inputWidth, inputHeight));

		// Prepare input to nework
		cv::Mat resizedImage, preprocessedImage;
		resizedImageRGB.convertTo(resizedImage, CV_32F);

		tf->model->PrepareInputToNetwork(resizedImage, preprocessedImage);

		tf->model->LoadInputToTensor(preprocessedImage, cv::Size(inputWidth, inputHeight), tf->inputTensorValues);

		// Run network inference
		tf->model->RunNetworkInference(
			tf->session,
			tf->inputNames,
			tf->outputNames,
			tf->inputTensor,
			tf->outputTensor
		);

		// Get output
		// Map network output to cv::Mat
		cv::Mat outputImage = tf->model->GetNetworkOutput(
			Model::Activation::NONE,
			tf->outputDims,
			tf->outputTensorValues
		);

		// Assign output to input in some models that have temporal information
		tf->model->AssignOutputToInput(tf->outputTensorValues, tf->inputTensorValues);

		// Post-process output. The image will now be in [0,1] float, BHWC format
		tf->model->PostprocessOutput(outputImage);

		// Convert [0,1] float to CV_8U [0,255]
		outputImage.convertTo(output, CV_8U, 255.0);

		return true;
	}

}
