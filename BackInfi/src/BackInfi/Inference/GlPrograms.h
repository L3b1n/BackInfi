#ifndef CONSTS_H
#define CONSTS_H

#include <string>

const char* const MODEL_SINET = "SINet_Softmax_simple";
const char* const MODEL_MEDIAPIPE = "mediapipe";
const char* const MODEL_SELFIE = "selfie_segmentation";
const char* const MODEL_RVM = "rvm_mobilenetv3_fp32";
const char* const MODEL_PPHUMANSEG = "pphumanseg_fp32";
const char* const MODEL_ENHANCE_TBEFN = "tbefn_fp32";
const char* const MODEL_ENHANCE_URETINEX = "uretinex_net_180x320";
const char* const MODEL_ENHANCE_SGLLIE = "semantic_guided_llie_180x324";
const char* const MODEL_ENHANCE_ZERODCE = "zero_dce_180x320";

const char* const USEGPU_CPU = "cpu";
const char* const USEGPU_DML = "dml";
const char* const USEGPU_CUDA = "cuda";
const char* const USEGPU_TENSORRT = "tensorrt";
const char* const USEGPU_COREML = "coreml";

const char* const EFFECT_PATH = "effects/mask_alpha_filter.effect";
const char* const KAWASE_BLUR_EFFECT_PATH = "effects/kawase_blur.effect";
const char* const BLEND_EFFECT_PATH = "effects/blend_images.effect";

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

		vec3 outputRGB;
		float a = (1.0 - maskTexture.r) * inputRGBA.a;
		// Because of output type I want to get back
		outputRGB.bgr = inputRGBA.rgb * a + texture(frame2, sample_coordinate).bgr * (1.0 - a);
		frag_out = outputRGB;
	}
)";

#endif /* CONSTS_H */