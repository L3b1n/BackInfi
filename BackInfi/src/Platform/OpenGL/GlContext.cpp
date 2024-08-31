#include "bcpch.h"

#include "Platform/OpenGL/GlContext.h"

#include <GLFW/glfw3.h>
#include <glad/glad.h>

namespace BackInfi
{

	GlContext::GlContext(GLFWwindow* windowHandle)
	{
		m_WindowHandle = windowHandle;

		BC_CORE_ASSERT(m_WindowHandle, "Window handle is null!");
	}

	void GlContext::Init()
	{
		glfwMakeContextCurrent(m_WindowHandle);
		int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		BC_CORE_ASSERT(status, "Failed to initialize Glad!");
		
		BC_CORE_INFO("OpenGL Info:");
		BC_CORE_INFO("  Vendor:   {0}", (const char*)glGetString(GL_VENDOR));
		BC_CORE_INFO("  Renderer: {0}", (const char*)glGetString(GL_RENDERER));
		BC_CORE_INFO("  Version:  {0}", (const char*)glGetString(GL_VERSION));
		
		BC_CORE_ASSERT(GLVersion.major > 4 || (GLVersion.major == 4 && GLVersion.minor >= 5), "Hazel requires at least OpenGL version 4.5!");
	}

	void GlContext::SwapBuffers()
	{
		glfwSwapBuffers(m_WindowHandle);
	}

}
