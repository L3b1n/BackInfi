#include "bcpch.h"
#include "RendererAPI.h"

namespace BackInfi
{
	RendererAPI::API RendererAPI::s_API = RendererAPI::API::OPENGL;

	std::unique_ptr<RendererAPI> RendererAPI::Create()
	{
		return std::unique_ptr<RendererAPI>();
	}

}
