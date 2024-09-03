#include "bcpch.h"

#include "BackInfi/Renderer/Renderer.h"
#include "BackInfi/Renderer/FrameBuffer.h"

#include "Platform/OpenGL/GlFrameBuffer.h"

namespace BackInfi
{

	std::shared_ptr<FrameBuffer> FrameBuffer::Create(const FrameBufferSpecs& specs)
	{
		switch (RendererAPI::GetAPI())
		{
		case RendererAPI::API::NONE:
			BC_CORE_ASSERT(false, "RendererAPI::NONE is currently not supported!"); return nullptr;
		case RendererAPI::API::OPENGL:
			return std::make_shared<GlFrameBuffer>(specs);
		}

		return nullptr;
	}

}