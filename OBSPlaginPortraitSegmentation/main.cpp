#include <iostream>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <BackgroundFilter.h>
#include <opencv2/opencv.hpp>

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void on_trackbar(int, void*) {
    cv::updateWindow("test");
}

void drawGLTexture(GLFWwindow* window) {

    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 1);
    glVertex2f(-1, -1);

    glTexCoord2f(1, 1);
    glVertex2f(1, -1);

    glTexCoord2f(0, 0);
    glVertex2f(-1, 1);

    glTexCoord2f(1, 1);
    glVertex2f(1, -1);

    glTexCoord2f(1, 0);
    glVertex2f(1, 1);

    glTexCoord2f(0, 0);
    glVertex2f(-1, 1);
    glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();

    glFlush();
    glFinish();

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

const int width = 1280;
const int height = 720;

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

int main() {
    // glfwInit();
    // //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // uncomment this for windowsless glfw
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // // glfw window creation
    // // --------------------
    // GLFWwindow* window = glfwCreateWindow(width, height, "LearnOpenGL", NULL, NULL);
    // if (window == NULL)
    // {
    //     std::cout << "Failed to create GLFW window" << std::endl;
    //     glfwTerminate();
    //     return -1;
    // }
    // glfwMakeContextCurrent(window);
    // glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // // glad: load all OpenGL function pointers
    // // ---------------------------------------
    // if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    // {
    //     std::cout << "Failed to initialize GLAD" << std::endl;
    //     return -1;
    // }

    // std::cout << "GLTest:" << glGetString(GL_VERSION) << "\n";

    // To make sure the program does not quit running
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FPS, 30);

    cv::namedWindow("Background remove app", cv::WINDOW_NORMAL);
    cv::resizeWindow("Background remove app", width, height);

    BackgroundFilter filter(height, width);
    filter.filterUpdate(true);
    filter.filterActivate();
    filter.loadBackground("background.jpg");

    // const auto mask_channel = 1;
    // filter.GlSetup(mask_channel);

    //glDisable(GL_BLEND);

    int frames = 0;
    double fps = 0.0;
    float seconds = 0.0;
    bool load_flag = true;
    auto start_time = std::chrono::high_resolution_clock::now();

    //int rotx = 0, roty = 0;
    //cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    //cv::createTrackbar("X-rotation", "test", &rotx, 10, on_trackbar);
    //cv::createTrackbar("Y-rotation", "test", &roty, 10, on_trackbar);

    while (load_filter) {
        cv::Mat temp;
        cap >> temp;

        // processInput(window);

        filter.filterVideoTick(temp.rows, temp.cols, temp.type(), temp.data);
        filter.blendSegmentationSmoothing(0.98);
        cv::Mat frame = filter.blendBackgroundAndForeground();
        //unsigned char* data = filter.blendBackgroundAndForeground();
        
        //unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * 1920 * 1080 * 4);
        //filter.blendBackgroundAndForeground(data);
        //cv::Mat frame(temp.size(), temp.type(), data);

        // glfwSwapBuffers(window);
        // glfwPollEvents();

        //drawGLTexture(window);

        frames++;
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
        if (elapsed_time >= 1.0) {
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
    // glfwTerminate();
    return 0;
}