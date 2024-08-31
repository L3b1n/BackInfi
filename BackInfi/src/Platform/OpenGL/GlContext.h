#pragma once

#include "BackInfi/Renderer/GraphicsContext.h"

struct GLFWwindow;

namespace BackInfi
{

	class GlContext : public GraphicsContext
	{
	public:
		GlContext(GLFWwindow* windowHandle);

		virtual void Init() override;
		virtual void SwapBuffers() override;
	private:
		GLFWwindow* m_WindowHandle;
	};

}