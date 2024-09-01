#include "bcpch.h"

#include "BackInfi/Renderer/VertexArray.h"
#include "BackInfi/Renderer/RendererAPI.h"

#include "Platform/OpenGL/GlVertexArray.h"

namespace BackInfi
{

	std::shared_ptr<VertexArray> VertexArray::Create()
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlVertexArray>();
		}

		return nullptr;
	}

}
