#include "bcpch.h"

#include "BackInfi/Renderer/RendererAPI.h"

#include "Platform/Windows/WindowsWindow.h"

namespace BackInfi
{

	static uint8_t s_GLFWWindowCount = 0;

	static void GLFWErrorCallback(int error, const char* description)
	{
		BC_CORE_ERROR("GLFW Error ({0}): {1}", error, description);
	}

	WindowsWindow::WindowsWindow(const WindowProp& prop)
	{
		Init(prop);
	}

	WindowsWindow::~WindowsWindow()
	{
		ShutDown();
	}

	void WindowsWindow::Init(const WindowProp& prop)
	{
		m_Info.Title  = prop.Title;
		m_Info.Width  = prop.Width;
		m_Info.Height = prop.Height;

		BC_CORE_INFO("Creating window {0} ({1}, {2})", m_Info.Title, m_Info.Width, m_Info.Height);

		if (s_GLFWWindowCount == 0)
		{
			int success = glfwInit();
			BC_CORE_ASSERT(success, "Could not initialize GLFW!");
			glfwSetErrorCallback(GLFWErrorCallback);
		}

		
		#if defined(BC_DEBUG)
		if (RendererAPI::GetAPI() == RendererAPI::API::OPENGL)
			glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
		#endif
		m_Window = glfwCreateWindow((int)m_Info.Width, (int)m_Info.Height, m_Info.Title.c_str(), nullptr, nullptr);
		++s_GLFWWindowCount;

		m_Context = GraphicsContext::Create(m_Window);
		m_Context->Init();

		glfwSetWindowUserPointer(m_Window, &m_Info);
		SetVSync(true);

		glfwSetWindowSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);
			info.Width  = width;
			info.Height = height;

			//WindowResizeEvent event(width, height);
			//info.EventCallback(event);
		});

		glfwSetWindowCloseCallback(m_Window, [](GLFWwindow* window)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);
			//WindowCloseEvent event;
			//info.EventCallback(event);
		});
	}

	void WindowsWindow::ShutDown()
	{
		glfwDestroyWindow(m_Window);
		--s_GLFWWindowCount;

		if (s_GLFWWindowCount == 0)
		{
			glfwTerminate();
		}
	}

	void WindowsWindow::OnUpdate()
	{
		glfwPollEvents();
		m_Context->SwapBuffers();
	}

	void WindowsWindow::SetVSync(bool enabled)
	{
		if (enabled)
			glfwSwapInterval(1);
		else
			glfwSwapInterval(0);

		m_Info.VSync = enabled;
	}

	bool WindowsWindow::IsVSync() const
	{
		return m_Info.VSync;
	}

}