#include "bcpch.h"

#include "BackInfi/Renderer/RendererAPI.h"
#include "BackInfi/Renderer/GraphicsContext.h"

#include "Platform/OpenGL/GlContext.h"

namespace BackInfi
{

	std::unique_ptr<GraphicsContext> GraphicsContext::Create(void* window)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_unique<GlContext>(static_cast<GLFWwindow*>(window));
		}

		return nullptr;
	}

}