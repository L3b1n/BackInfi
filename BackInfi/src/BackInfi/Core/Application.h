#pragma once

#include "BackInfi/Core/Base.h"
#include "BackInfi/Core/Layer.h"
#include "BackInfi/Core/Window.h"

#include "BackInfi/ImGui/ImGuiLayerGLFWOpenGL.h"

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

		void Close();

		Window& GetWindow() { return *m_Window; }
		static Application& Get() { return *s_Instance; }

		void Run();
	private:
		bool OnWindowClose(WindowCloseEvent& e);
		bool OnWindowResize(WindowResizeEvent& e);

	private:
		bool m_Running;
		float m_LastFrameTime;
		std::unique_ptr<Window> m_Window;
		LayerStack m_LayerStack;
		ImGuiLayerGLFWOpenGL* m_ImGuiLayer;

	private:
		static Application* s_Instance;
	};

}