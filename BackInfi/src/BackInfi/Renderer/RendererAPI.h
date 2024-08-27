#pragma once
#include <glm/glm.hpp>

namespace BackInfi
{
	class RendererAPI
	{
	public:
		enum class API
		{
			NONE   = 0,
			OPENGL = 1
		};

	public:
		virtual ~RendererAPI() = default;

		virtual void Init() = 0;

		static API GetAPI() { return s_API; }
		static std::unique_ptr<RendererAPI> Create();

	private:
		static API s_API;
	};
}