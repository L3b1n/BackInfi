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



#endif /* CONSTS_H */