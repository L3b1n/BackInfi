#include "bcpch.h"

#include "BackInfi/Renderer/Renderer.h"

#include "Platform/OpenGL/GlRendererAPI.h"

namespace BackInfi
{
	// RendererAPI
	// ------------------------------------------------------------------------
	RendererAPI::API RendererAPI::s_API = RendererAPI::API::OPENGL;

	std::unique_ptr<RendererAPI> RendererAPI::Create()
	{
		switch (s_API)
		{
		case RendererAPI::API::NONE:    BC_CORE_ASSERT(false, "RendererAPI::None is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:  return std::make_unique<GlRendererAPI>();
		}

		return nullptr;
	}

	// Renderer
	// ------------------------------------------------------------------------
	void Renderer::Init()
	{
		BC_PROFILE_FUNC();

		RenderCommand::Init();
	}

	void Renderer::Shutdown()
	{

	}

	void Renderer::OnWindowResize(uint32_t width, uint32_t height)
	{
		RenderCommand::SetViewport(0, 0, width, height);
	}

	void Renderer::BeginScene()
	{
	}

	void Renderer::EndScene()
	{
	}

	void Renderer::Submit(const std::shared_ptr<Shader>& shader, const std::shared_ptr<VertexArray>& vertexArray, const glm::mat4& transform)
	{
		shader->Bind();
		shader->SetMat4("transform", transform);

		vertexArray->Bind();
		RenderCommand::DrawIndexed(vertexArray);
	}

	// RenderCommand
	// ------------------------------------------------------------------------
	std::unique_ptr<RendererAPI> RenderCommand::s_RendererAPI = RendererAPI::Create();

}
