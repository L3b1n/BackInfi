# BackInfi
BackInfi is primarily an early-stage interactive application and rendering pipline that allows you to replace the background in portrait images and video on Windows. It is created by using ONNX Runtime inference and OpenGL (glad) post-processing rendering. 
For now implementation is based on five models:
* [MediaPipe](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/082_MediaPipe_Meet_Segmentation);
* [Selfie segmentation](https://drive.google.com/file/d/1dCfozqknMa068vVsO2j_1FgZkW_e3VWv/preview);
* [SINet_softmax_simple](https://github.com/anilsathyan7/Portrait-Segmentation/tree/master/SINet);
* [pphumanseg_fp32](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/contrib/PP-HumanSeg);
* [rvm_mobilnetv3_fp32](https://github.com/PeterL1n/RobustVideoMatting).

## Getting started
Visual studio 2019 or 2022 is recommended. BackInfi is untested on other development environments.

### 1. Downloading the repository:
Start by cloning the repository with `git clone --recursive https://github.com/L3b1n/BackInfi`.

If the repository was cloned non-recursively previously, use `git submodule update --init` to clone the necessary submodules.

### 2. Configuring and dependencies:
 * All of the needed dependencies are install with main repository.
 * So just run the [WinGenProjetcs.bat](./scripts/WinGenProjects.bat) script file. It will automatically generate a Visual Studio solution file for user's usage.

## Plans
 * Separate OpenCV and OpenGL versions
 * Add background blurring mode
 * Add stats of the renderer (fps, inference time,...)
 * Separate WinAPI and GLFW on Windows
 * Support for Mac, Linux and Windows
    * Native rendering API support (DirectX, Vulkan)

## Dependencies
 * [glm](https://github.com/g-truc/glm)
 * [glad](https://glad.dav1d.de)
 * [GLFW](https://github.com/glfw/glfw)
 * [ImGui](https://github.com/ocornut/imgui)
 * [spdlog](https://github.com/gabime/spdlog)
 * [OpenCV](https://github.com/opencv/opencv)
 * [OnnxRuntime](https://github.com/microsoft/onnxruntime) and [OnnxRutnime static lib](https://github.com/csukuangfj/onnxruntime-build.git)
