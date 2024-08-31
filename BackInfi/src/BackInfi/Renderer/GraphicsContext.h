#pragma once

#include "BackInfi/Core/Base.h"

namespace BackInfi
{

	class GraphicsContext
	{
	public:
		virtual ~GraphicsContext() = default;

		virtual void Init() = 0;
		virtual void SwapBuffers() = 0;

		static std::unique_ptr<GraphicsContext> Create(void* window);
	};

}