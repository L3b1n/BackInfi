#ifndef ORT_SESSION_UTILS_H
#define ORT_SESSION_UTILS_H

#include <opencv2/core/types.hpp>

#include "../FilterData.h"

void createOrtSession(BackgroundRemovalFilter *tf);

bool runFilterModelInference(BackgroundRemovalFilter *tf, const cv::Mat &imageBGRA, cv::Mat &output);

#endif /* ORT_SESSION_UTILS_H */
