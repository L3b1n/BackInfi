#include "bcpch.h"
#include "BackgroundFilter.h"
#include "ort-utils/ort-session-utils.h"
#include <opencv2/opencv.hpp>

void OpenGlClError() {
    while (glGetError());
}
bool OpenGlError(const char* func, int line, const char* file) {
    while (GLenum error = glGetError()) {
        std::cout << "GLTest: OPENGL ERROR: " << error << " in function " << func << " at line " << line << ", FILE: " << file << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(10000));
        return false;
    }
    return true;
}

#define ASSERT(x) if(!(x)) __debugbreak();
#define CALL(x) OpenGlClError(); x; ASSERT(OpenGlError(#x, __LINE__, __FILE__));

#if DEBUG
#define GL_DEBUG_LOG(type, object, action)                        \
  do {                                                            \
    GLint log_length = 0;                                         \
    glGet##type##iv(object, GL_INFO_LOG_LENGTH, &log_length);     \
    if (log_length > 0) {                                         \
      GLchar* log = static_cast<GLchar*>(malloc(log_length));     \
      glGet##type##InfoLog(object, log_length, &log_length, log); \
      LOG(INFO) << #type " " action " log:\n" << log;             \
      free(log);                                                  \
    }                                                             \
  } while (0)
#else
#define GL_DEBUG_LOG(type, object, action)
#endif

#if !defined(_STRINGIFY)
#define __STRINGIFY(_x) #_x
#define _STRINGIFY(_x) __STRINGIFY(_x)
#endif

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

#define GLES_VERSION_COMPAT "#version 330 \n"

#define PRECISION_COMPAT                              \
  GLES_VERSION_COMPAT                                 \
  "#ifdef GL_ES \n"                                   \
  "#define DEFAULT_PRECISION(p, t) precision p t; \n" \
  "#else \n"                                          \
  "#define DEFAULT_PRECISION(p, t) \n"                \
  "#define lowp \n"                                   \
  "#define mediump \n"                                \
  "#define highp \n"                                  \
  "#endif  // defined(GL_ES) \n"

#define VERTEX_PREAMBLE     \
  PRECISION_COMPAT          \
  "#if __VERSION__ < 130\n" \
  "#define in attribute\n"  \
  "#define out varying\n"   \
  "#endif  // __VERSION__ < 130\n"

// Note: on systems where highp precision for floats is not supported (look up
// GL_FRAGMENT_PRECISION_HIGH), we replace it with mediump.
// gl_FragColor is re-defined to 'frag_out' when using 3.20 Core (GLSL 330+).
#define FRAGMENT_PREAMBLE                                        \
  PRECISION_COMPAT                                               \
  "#if __VERSION__ < 130\n"                                      \
  "#define in varying\n"                                         \
  "#define texture texture2D\n"                                  \
  "#if defined(GL_ES) && !defined(GL_FRAGMENT_PRECISION_HIGH)\n" \
  "#define highp mediump\n"                                      \
  "#endif  // GL_ES && !GL_FRAGMENT_PRECISION_HIGH\n"            \
  "#elif __VERSION__ > 320 && !defined(GL_ES)\n"                 \
  "out vec4 frag_out; \n"                                        \
  "#define gl_FragColor frag_out\n"                              \
  "#define texture2D texture\n"                                  \
  "#endif  // __VERSION__ < 130\n"

const GLchar* const kMediaPipeFragmentShaderPreamble = FRAGMENT_PREAMBLE;

const GLchar* const kBasicVertexShader = VERTEX_PREAMBLE _STRINGIFY(
    // vertex position in clip space (-1..1)
    in vec4 position;

    // texture coordinate for each vertex in normalized texture space (0..1)
    in mediump vec4 texture_coordinate;

    // texture coordinate for fragment shader (will be interpolated)
    out mediump vec2 sample_coordinate;

    void main() {
        gl_Position = position;
        sample_coordinate = texture_coordinate.xy;
    }
);

constexpr int kMaxShaderInfoLength = 1024;

GLint GlhCompileShader(GLenum target, const GLchar* source, GLuint* shader,
    bool force_log_errors) {
    *shader = glCreateShader(target);
    if (*shader == 0) {
        return GL_FALSE;
    }
    CALL(glShaderSource(*shader, 1, &source, NULL));
    CALL(glCompileShader(*shader));

    GL_DEBUG_LOG(Shader, *shader, "compile");

#if UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING
    if (!force_log_errors) {
        return GL_TRUE;
    }
#endif  // UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING

    GLint status;

    CALL(glGetShaderiv(*shader, GL_COMPILE_STATUS, &status));
    //LOG_IF(ERROR, status == GL_FALSE) << "Failed to compile shader:\n" << source;

    if (status == GL_FALSE) {
        int length = 0;
        GLchar cmessage[kMaxShaderInfoLength];
        glGetShaderInfoLog(*shader, kMaxShaderInfoLength, &length, cmessage);
        std::cerr << "Error message: " << std::string(cmessage, length);
    }
    return status;
}

GLint GlhLinkProgram(GLuint program, bool force_log_errors) {
    CALL(glLinkProgram(program));

#if UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING
    if (!force_log_errors) {
        return GL_TRUE;
    }
#endif  // UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING

    GLint status;

    GL_DEBUG_LOG(Program, program, "link");

    CALL(glGetProgramiv(program, GL_LINK_STATUS, &status));
    if (status == GL_FALSE) {
        std::cerr << "Failed to link program " << program << "\n";
    }

    return status;
}

GLint GlhCreateProgram(const GLchar* vert_src, const GLchar* frag_src,
    GLsizei attr_count, const GLchar* const* attr_names,
    const GLint* attr_locations, GLuint* program,
    bool force_log_errors) {
    GLuint vert_shader = 0;
    GLuint frag_shader = 0;
    GLint ok = GL_TRUE;

    *program = glCreateProgram();
    if (*program == 0) {
        return GL_FALSE;
    }

    ok = ok && GlhCompileShader(GL_VERTEX_SHADER, vert_src, &vert_shader,
        force_log_errors);
    ok = ok && GlhCompileShader(GL_FRAGMENT_SHADER, frag_src, &frag_shader,
        force_log_errors);

    if (ok) {
        CALL(glAttachShader(*program, vert_shader));
        CALL(glAttachShader(*program, frag_shader));

        // Attribute location binding must be set before linking.
        for (int i = 0; i < attr_count; i++) {
            CALL(glBindAttribLocation(*program, attr_locations[i], attr_names[i]));
        }

        ok = GlhLinkProgram(*program, force_log_errors);
    }

    if (vert_shader) CALL(glDeleteShader(vert_shader));
    if (frag_shader) CALL(glDeleteShader(frag_shader));

    if (!ok) {
        CALL(glDeleteProgram(*program));
        *program = 0;
    }

    return ok;
}

void BackgroundFilter::filterUpdate(const bool enableThreshold_, const float threshold_,
    const float contourFilter_, const float smoothContour_,
    const float feather_, const int maskEveryXframes_,
    const int64_t blurBackground_, const std::string useGpu_,
    const std::string model_, const uint32_t numThreads_) {

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
        tf->useGPU != newUseGpu || tf->numThreads != newNumThreads) {
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
            tf->model.reset(new ModelMediaPipe);
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

void BackgroundFilter::processImageForBackground(const cv::Mat& imageBGRA, cv::Mat& backgroundMask) {
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
    } else {
        backgroundMask = 255 - outputImage;
    }
}

void BackgroundFilter::filterVideoTick(const int height, const int width,
    const int type, unsigned char* data) {

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
    const double combine_with_previous_ratio_) {
    if (tf->backgroundMask.empty()) { std::cerr << "Error! Background mask is empty!\n"; }

    current_mask = tf->backgroundMask;
    if (previous_mask.empty()) { previous_mask = tf->backgroundMask; }

    if (previous_mask.type() != current_mask.type()) { 
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

cv::Mat BackgroundFilter::TextureToMat(GLuint texture_id) {
    CALL(glBindTexture(GL_TEXTURE_2D, texture_id));
    unsigned char* texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 4);
    CALL(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_bytes));

    return cv::Mat(tf->inputRGBA.size(), CV_8UC4, texture_bytes); // CV_8UC3 for simple image, ..C4 for mask
}

cv::Mat BackgroundFilter::BufferToMat() {
    cv::Mat output_mat(tf->inputRGBA.size(), CV_8UC4); // because of RGBA
    // use fast 4-byte alignment (default anyway) if possible
    CALL(glPixelStorei(GL_PACK_ALIGNMENT, (output_mat.step & 3) ? 1 : 4));

    // set length of one complete row in destination data (doesn't need to equal img.cols)
    CALL(glPixelStorei(GL_PACK_ROW_LENGTH, (GLint)(output_mat.step / output_mat.elemSize())));
    CALL(glReadPixels(0, 0, output_mat.cols, output_mat.rows, GL_RGBA, GL_UNSIGNED_BYTE, output_mat.data));
    cv::flip(output_mat, output_mat, 0);
    cv::cvtColor(output_mat, output_mat, cv::COLOR_RGBA2BGR);

    return output_mat;
}

void BackgroundFilter::BindCVMatToGLTexture(cv::Mat& image, GLuint& imageTexture) {
    if (image.empty()) {
        std::cerr << "image empty\n";
    }
    else {
        //CALL(glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE));
        //CALL(glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE));
        CALL(glGenTextures(1, &imageTexture));
        CALL(glBindTexture(GL_TEXTURE_2D, imageTexture));

        // Set texture clamping method
        //CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)); // I have those lines, but in mediapipe realisation they have lines below
        //CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)); // I have those lines, but in mediapipe realisation they have lines below
        CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        
        //CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); // I have this line, but in mediapipe realisation they have line below
        CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));


        //cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        CALL(
          glTexImage2D(
            GL_TEXTURE_2D,       // Type of texture
            0,                   // Pyramid level (for mip-mapping) - 0 is the top level
            GL_RGBA,             // Internal colour format to convert to
            image.cols,          // Image width  i.e. 640 for Kinect in standard mode
            image.rows,          // Image height i.e. 480 for Kinect in standard mode
            0,                   // Border width in pixels (can either be 1 or 0)
            GL_RGBA,             // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
            GL_UNSIGNED_BYTE,    // Image data type
            image.ptr()          // The actual image data itself
          ));
        CALL(glBindTexture(GL_TEXTURE_2D, 0));
    }
}

bool BackgroundFilter::GlSetup(const int mask_channel) {
    // Load vertex and fragment shaders
    const GLint attr_location[NUM_ATTRIBUTES] = {
        ATTRIB_VERTEX,
        ATTRIB_TEXTURE_POSITION,
    };
    const GLchar* attr_name[NUM_ATTRIBUTES] = {
        "position",
        "texture_coordinate",
    };

    std::string mask_component;
    switch (mask_channel) {
    case 0:
    case 1:
        mask_component = "r";
        break;
    case 2:
        mask_component = "a";
        break;
    }

    const std::string frag_src_tex =
        std::string(kMediaPipeFragmentShaderPreamble) +
        R"(
    DEFAULT_PRECISION(highp, float)

    in vec2 sample_coordinate;
    uniform sampler2D frame1;
    uniform sampler2D frame2;
    uniform sampler2D mask;

    void main() {
      vec4 color1 = texture2D(frame1, sample_coordinate);
      vec4 color2 = texture2D(frame2, sample_coordinate);
      vec4 weight = texture2D(mask, sample_coordinate);

    #define MASK_COMPONENT )" +
        mask_component +
        R"(

      gl_FragColor = mix(color1, color2, weight.MASK_COMPONENT);
    }
  )";

    const GLchar* frag_src_const = R"(
    precision highp float;

    varying vec2 sample_coordinate;
    uniform sampler2D frame1;
    uniform sampler2D frame2;
    uniform float mask;

    void main() {
      vec4 color1 = texture2D(frame1, sample_coordinate);
      vec4 color2 = texture2D(frame2, sample_coordinate);
      float weight = mask;

      gl_FragColor = mix(color1, color2, weight);
    }
  )";

    // shader program
    CALL(GlhCreateProgram(kBasicVertexShader,
        true ? frag_src_tex.c_str() : frag_src_const,
        NUM_ATTRIBUTES, &attr_name[0], attr_location, &program_, false));
    if (!program_) std::cerr << "Problem initializing the program.\n";
    CALL(unif_frame1_ = glGetUniformLocation(program_, "frame1"));
    CALL(unif_frame2_ = glGetUniformLocation(program_, "frame2"));
    CALL(unif_mask_ = glGetUniformLocation(program_, "mask"));
    return true; 
}

bool BackgroundFilter::GlRender(const float mask_const) {
    CALL(glUseProgram(program_));
    CALL(glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, kBasicSquareVertices));
    CALL(glEnableVertexAttribArray(ATTRIB_VERTEX));
    CALL(glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0,
       kBasicTextureVertices));
    CALL(glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION));

    CALL(glUniform1i(unif_frame1_, 1));
    CALL(glUniform1i(unif_frame2_, 2));
    if (true) {
        CALL(glUniform1i(unif_mask_, 3));
    } else
        glUniform1f(unif_mask_, mask_const);

    CALL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
    return true;
}

cv::Mat BackgroundFilter::blendBackgroundAndForeground() {

    if (tf->inputRGBA.empty()) { std::cerr << "Error! Input image is empty!\n"; }

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

    cv::Mat mask_matSingle = tf->backgroundMask;

    //cv::imshow("InputRGBA", mask_matSingle);

    // opencv part --------------------------------------------------
    
    cv::Mat mask_mat = tf->backgroundMask;
    if (mask_mat.channels() > 1) {
       std::vector<cv::Mat> channels;
       cv::split(mask_mat, channels);
       mask_mat = channels[0];
    }
    
    cv::Mat image;
    cv::cvtColor(imageRGBA, image, cv::COLOR_BGRA2BGR);
    cv::Mat output_mat(image.rows, image.cols, image.type());

    const int invert_mask = false ? 1 : 0;
    const int adjust_with_luminance = false ? 1 : 0;

    cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range& range) {
       for (int i = range.start; i < range.end; i++) {
           for (int j = 0; j < image.cols; j++) {
               const float weight = mask_mat.at<float>(i, j) * (1.0 / 255.0);
               output_mat.at<cv::Vec3b>(i, j) =
                   Blend(image.at<cv::Vec3b>(i, j), recolor, weight, invert_mask,
                       adjust_with_luminance);
           }
       }
       });

    cv::flip(output_mat, output_mat, 0);
    cv::cvtColor(output_mat, output_mat, cv::COLOR_BGR2RGB);
    
    // end of opencv part -------------------------------------------


    // testing opengl part ------------------------------------------
    // cv::Mat mask_mat;
    // mask_matSingle.convertTo(mask_matSingle, CV_8UC1);
    // cv::Mat zeros = cv::Mat::zeros(mask_matSingle.size(), mask_matSingle.type());
    // std::vector<cv::Mat> testVector = { mask_matSingle, zeros, zeros, zeros };
    // cv::merge(testVector, mask_mat);

    // // now flip is in filterVideoTick

    // GLuint target_mask;
    // GLuint target_input;
    // GLuint target_background;
    // GLuint framebuffer_ = 0;
    // GLuint output;

    // // version opengl 4.6, but i have metal 3.3
    // CALL(glDisable(GL_BLEND));

    // //CALL(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_));
    // //CALL(glViewport(0, 0, width, height));

    // //CALL(glActiveTexture(GL_TEXTURE0));
    // //CALL(glBindTexture(GL_TEXTURE_2D, output));
    // //CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
    // //    output, 0));

    // BindCVMatToGLTexture(imageRGBA, target_input);
    // BindCVMatToGLTexture(mask_mat, target_mask);
    // BindCVMatToGLTexture(background, target_background);

    // CALL(glActiveTexture(GL_TEXTURE1));
    // CALL(glBindTexture(GL_TEXTURE_2D, target_input));

    // CALL(glActiveTexture(GL_TEXTURE2));
    // CALL(glBindTexture(GL_TEXTURE_2D, target_background));

    // // use mask, not const mask
    // const float mask_const = -1;

    // CALL(glActiveTexture(GL_TEXTURE3));
    // CALL(glBindTexture(GL_TEXTURE_2D, target_mask));

    // GlRender(mask_const);

    // CALL(glActiveTexture(GL_TEXTURE3));
    // CALL(glBindTexture(GL_TEXTURE_2D, 0));
    // //

    // CALL(glActiveTexture(GL_TEXTURE2));
    // CALL(glBindTexture(GL_TEXTURE_2D, 0));

    // CALL(glActiveTexture(GL_TEXTURE1));
    // CALL(glBindTexture(GL_TEXTURE_2D, 0));
    // CALL(glFlush());
    
    // //glEnable(GL_TEXTURE_2D);
    // //glBindTexture(GL_TEXTURE_2D, target_mask);

    // //cv::Mat output_mat = TextureToMat(target_mask);
    // cv::Mat output_mat = BufferToMat();
    // end of testing opengl part -----------------------------------

    return output_mat;
}


//#version 330
//
//#ifdef GL_ES
//#define DEFAULT_PRECISION(p, t) precision p t;
//#else
//#define DEFAULT_PRECISION(p, t)
//#define lowp
//#define mediump
//#define highp
//#endif  // defined(GL_ES)
//
//#if __VERSION__ < 130
//#define in attribute
//#define out varying
//#endif  // __VERSION__ < 130
//
//    // vertex position in clip space (-1..1)
//    in vec4 position;
//
//// texture coordinate for each vertex in normalized texture space (0..1)
//in mediump vec4 texture_coordinate;
//
//// texture coordinate for fragment shader (will be interpolated)
//out mediump vec2 sample_coordinate;
//
//void main() {
//    gl_Position = position;
//    sample_coordinate = texture_coordinate.xy;
//}

//#version 330
//
//#ifdef GL_ES
//#define DEFAULT_PRECISION(p, t) precision p t;
//#else
//#define DEFAULT_PRECISION(p, t)
//#define lowp
//#define mediump
//#define highp
//#endif  // defined(GL_ES)
//
//#if __VERSION__ < 130                                     
//#define in varying
//#define texture texture2D
//#if defined(GL_ES) && !defined(GL_FRAGMENT_PRECISION_HIGH)
//#define highp mediump
//#endif  // GL_ES && !GL_FRAGMENT_PRECISION_HIGH
//#elif __VERSION__ > 320 && !defined(GL_ES)
//out vec4 frag_out;
//#define gl_FragColor frag_out
//#define texture2D texture
//#endif  // __VERSION__ < 130
//
//DEFAULT_PRECISION(highp, float)
//
//in vec2 sample_coordinate;
//uniform sampler2D frame1;
//uniform sampler2D frame2;
//uniform sampler2D mask;
//
//void main() {
//    vec4 color1 = texture2D(frame1, sample_coordinate);
//    vec4 color2 = texture2D(frame2, sample_coordinate);
//    vec4 weight = texture2D(mask, sample_coordinate);
//
//#define MASK_COMPONENT r
//
//    gl_FragColor = mix(color1, color2, weight.MASK_COMPONENT);
//}