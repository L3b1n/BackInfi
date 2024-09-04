#pragma once

#include "BackInfi/Core/Base.h"

#include "BackInfi/Events/Event.h"

namespace BackInfi
{

	struct WindowProp
	{
		std::string Title;
		uint32_t    Width;
		uint32_t    Height;

		WindowProp(
			const std::string& title = "BackInfi app",
			uint32_t width = 1280,
			uint32_t height = 720)
		{
			Title  = title;
			Width  = width;
			Height = height;
		}
	};

	class Window
	{
	public:
		virtual ~Window() = default;

		virtual void OnUpdate() = 0;

		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;

		virtual void SetEventCallback(const std::function<void(Event&)>& callback) = 0;
		virtual void SetVSync(bool enabled) = 0;
		virtual bool IsVSync() const = 0;

		virtual void* GetNativeWindow() const = 0;

		static std::unique_ptr<Window> Create(const WindowProp& props = WindowProp());
	};

}