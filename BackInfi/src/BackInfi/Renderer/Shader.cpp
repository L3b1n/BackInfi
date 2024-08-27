#include "bcpch.h"
#include "Shader.h"

#include "BackInfi/Renderer/RendererAPI.h"

#include "Platform/OpenGL/GlShader.h"

namespace BackInfi
{

	std::shared_ptr<Shader> Shader::Create(const std::string& filepath)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlShader>(filepath);
		}

		return nullptr;
	}

	std::shared_ptr<Shader> Shader::Create(
		const std::string& name,
		const std::string& vertexSrc,
		const std::string& fragmentSrc)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlShader>(name, vertexSrc, fragmentSrc);
		}

		return nullptr;
	}

}

