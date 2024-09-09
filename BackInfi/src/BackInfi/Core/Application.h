#pragma once

#include "BackInfi/Core/Base.h"
#include "BackInfi/Core/Layer.h"
#include "BackInfi/Core/Window.h"

#include "BackInfi/Renderer/Renderer.h"

#include "BackInfi/Events/WindowEvent.h"

namespace BackInfi
{

	class Application
	{
	public:
		Application();
		virtual ~Application();

		void OnEvent(Event& e);

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* layer);

		Window& GetWindow() { return *m_Window; }
		static Application& Get() { return *s_Instance; }

		void Run();
	private:
		bool OnWindowClose(WindowCloseEvent& e);
		bool OnWindowResize(WindowResizeEvent& e);

	private:
		bool m_Running;
		float m_LastFrameTime;
		LayerStack m_LayerStack;
		std::unique_ptr<Window> m_Window;

	private:
		static Application* s_Instance;
		//friend int ::main();
	};

}