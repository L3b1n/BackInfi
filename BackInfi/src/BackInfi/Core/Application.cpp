#include "bcpch.h"

#include "BackInfi/Core/Application.h"

#include "BackInfi/Utils/PlatformUtils.h"

namespace BackInfi
{

	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		BC_PROFILE_FUNC();

		m_Running       = true;
		m_LastFrameTime = 0.0f;

		BC_CORE_ASSERT(!s_Instance, "Application already exists!");
		s_Instance = this;

		m_Window = Window::Create(WindowProp("BackInfi app"));
		m_Window->SetEventCallback(BC_BIND_EVENT_FN(Application::OnEvent));

		Renderer::Init();

		m_ImGuiLayer = new ImGuiLayerGLFWOpenGL();
		PushOverlay(m_ImGuiLayer);
	}

	Application::~Application()
	{
		BC_PROFILE_FUNC();

		Renderer::Shutdown();
	}

	void Application::PushLayer(Layer* layer)
	{
		BC_PROFILE_FUNC();

		m_LayerStack.PushLayer(layer);
		layer->OnAttach();
	}

	void Application::PushOverlay(Layer* layer)
	{
		BC_PROFILE_FUNC();

		m_LayerStack.PushOverlay(layer);
		layer->OnAttach();
	}

	void Application::OnEvent(Event& e)
	{
		BC_PROFILE_FUNC();

		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(BC_BIND_EVENT_FN(OnWindowClose));
		dispatcher.Dispatch<WindowResizeEvent>(BC_BIND_EVENT_FN(OnWindowResize));

		for (auto it = m_LayerStack.rbegin(); it != m_LayerStack.rend(); ++it)
		{
			if (e.m_Handled)
				break;
			(*it)->OnEvent(e);
		}
	}

	void Application::Close()
	{
		m_Running = false;
	}

	void Application::Run()
	{
		BC_PROFILE_FUNC();

		while (m_Running)
		{
			BC_PROFILE_SCOPE("RunningLoop");
			float time = Time::GetTime();
			TimeStep ts = time - m_LastFrameTime;
			m_LastFrameTime = time;
			
			{
				BC_PROFILE_SCOPE("LayerStack OnUpdate");

				for (auto layer : m_LayerStack)
					layer->OnUpdate(ts);
			}

			m_ImGuiLayer->Begin();
			{
				BC_PROFILE_SCOPE("LayerStack OnImGuiRenderer");

				for (auto layer : m_LayerStack)
					layer->OnImGuiRender();
			}
			m_ImGuiLayer->End();

			m_Window->OnUpdate();
		}
	}

	bool Application::OnWindowClose(WindowCloseEvent& e)
	{
		m_Running = false;
		return true;
	}

	bool Application::OnWindowResize(WindowResizeEvent& e)
	{
		BC_PROFILE_FUNC();

		Renderer::OnWindowResize(e.GetWidth(), e.GetHeight());
		return true;
	}

}