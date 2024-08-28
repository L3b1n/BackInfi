#include "bcpch.h"

#include "BackInfi/Renderer/Texture.h"
#include "BackInfi/Renderer/RendererAPI.h"

#include "Platform/OpenGL/GlTexture.h"

namespace BackInfi
{

	std::shared_ptr<Texture2D> Texture2D::Create(const TextureSpecification& spec)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlTexture>(spec);
		}

		return nullptr;
	}

	std::shared_ptr<Texture2D> Texture2D::Create(const std::string& filepath)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlTexture>(filepath);
		}

		return nullptr;
	}

}
