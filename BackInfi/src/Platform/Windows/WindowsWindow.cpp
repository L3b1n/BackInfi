#include "bcpch.h"

#include "BackInfi/Renderer/Renderer.h"

#include "BackInfi/Events/KeyEvent.h"
#include "BackInfi/Events/MouseEvent.h"
#include "BackInfi/Events/WindowEvent.h"

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
		BC_PROFILE_FUNC();

		Init(prop);
	}

	WindowsWindow::~WindowsWindow()
	{
		BC_PROFILE_FUNC();

		ShutDown();
	}

	void WindowsWindow::Init(const WindowProp& prop)
	{
		BC_PROFILE_FUNC();

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

			WindowResizeEvent event(width, height);
			info.EventCallback(event);

			Renderer::OnWindowResize(width, height);
		});

		glfwSetWindowCloseCallback(m_Window, [](GLFWwindow* window)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);
			WindowCloseEvent event;
			info.EventCallback(event);
		});

		glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);

			switch (action)
			{
			case GLFW_PRESS: {
				KeyPressedEvent event(key, 0);
				info.EventCallback(event);
				break;
			}
			case GLFW_RELEASE: {
				KeyReleasedEvent event(key);
				info.EventCallback(event);
				break;
			}
			case GLFW_REPEAT: {
				KeyPressedEvent event(key, true);
				info.EventCallback(event);
				break;
			}
			}
		});

		glfwSetCharCallback(m_Window, [](GLFWwindow* window, unsigned int keycode)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);

			KeyTypedEvent event(keycode);
			info.EventCallback(event);
		});

		glfwSetMouseButtonCallback(m_Window, [](GLFWwindow* window, int button, int action, int mods)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);

			switch (action)
			{
			case GLFW_PRESS: {
				MouseButtonPressedEvent event(button);
				info.EventCallback(event);
				break;
			}
			case GLFW_RELEASE: {
				MouseButtonReleasedEvent event(button);
				info.EventCallback(event);
				break;
			}
			}
		});

		glfwSetScrollCallback(m_Window, [](GLFWwindow* window, double xOffset, double yOffset)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);

			MouseScrolledEvent event((float)xOffset, (float)yOffset);
			info.EventCallback(event);
		});

		glfwSetCursorPosCallback(m_Window, [](GLFWwindow* window, double xPos, double yPos)
		{
			WindowInfo& info = *(WindowInfo*)glfwGetWindowUserPointer(window);

			MouseMovedEvent event((float)xPos, (float)yPos);
			info.EventCallback(event);
		});
	}

	void WindowsWindow::ShutDown()
	{
		BC_PROFILE_FUNC();

		glfwDestroyWindow(m_Window);
		--s_GLFWWindowCount;

		if (s_GLFWWindowCount == 0)
		{
			glfwTerminate();
		}
	}

	void WindowsWindow::OnUpdate()
	{
		BC_PROFILE_FUNC();

		glfwPollEvents();
		m_Context->SwapBuffers();
	}

	void WindowsWindow::SetVSync(bool enabled)
	{
		BC_PROFILE_FUNC();

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