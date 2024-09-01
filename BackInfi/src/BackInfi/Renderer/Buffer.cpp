#include "bcpch.h"

#include "BackInfi/Renderer/Buffer.h"
#include "BackInfi/Renderer/RendererAPI.h"

#include "Platform/OpenGL/GlBuffer.h"

namespace BackInfi
{

	std::shared_ptr<VertexBuffer> VertexBuffer::Create(uint32_t size)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlVertexBuffer>(size);
		}

		return nullptr;
	}

	std::shared_ptr<VertexBuffer> VertexBuffer::Create(float* vertices, uint32_t size)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlVertexBuffer>(vertices, size);
		}

		return nullptr;
	}

	std::shared_ptr<IndexBuffer> IndexBuffer::Create(uint32_t* indices, uint32_t count)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlIndexBuffer>(indices, count);
		}

		return nullptr;
	}

}