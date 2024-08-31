#pragma once

#include "BackInfi/Core/Base.h"
#include "BackInfi/Core/Window.h"

#include "BackInfi/Renderer/GraphicsContext.h"

#include <GLFW/glfw3.h>

namespace BackInfi
{

	class WindowsWindow : public Window
	{
	public:
		WindowsWindow(const WindowProp& prop);
		virtual ~WindowsWindow();

		virtual void OnUpdate() override;

		virtual uint32_t GetWidth() const override { return m_Info.Width; }
		virtual uint32_t GetHeight() const override { return m_Info.Height; }

		//virtual void SetEventCallback(const std::function<void(void&)>& callback) override { m_Info.EventCallback = callback; }
		virtual void SetVSync(bool enabled) override;
		virtual bool IsVSync() const override;

		virtual void* GetNativeWindow() const override { return m_Window; }

	private:
		virtual void Init(const WindowProp& prop);
		virtual void ShutDown();

	private:
		GLFWwindow* m_Window;
		std::unique_ptr<GraphicsContext> m_Context;

		struct WindowInfo
		{
			std::string Title;
			unsigned int Width, Height;
			bool VSync;

			//std::function<void(void&)> EventCallback;
		};

		WindowInfo m_Info;
	};

}