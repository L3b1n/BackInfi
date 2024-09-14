#include "bcpch.h"

#include "BackInfi/Debug/GlDebug.h"

#include "Platform/OpenGL/GlRendererAPI.h"

#include <glad/glad.h>

namespace BackInfi
{

	void GlRendererAPI::Init()
	{
		BC_PROFILE_FUNC();

	#ifdef BC_DEBUG
		Debug::EnableGLDebugging();
	#endif

		// 4-byte pixel alignment
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
		glHint(GL_TEXTURE_COMPRESSION_HINT, GL_FASTEST);
		glHint(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, GL_NICEST);

		// SRGB output(even if input texture is 
		// non - sRGB->don't rely on texture used).
		// Your font is not using SRGB, for example
		// (not that it matters there, because no actual color is sampled from it).
		// But this could prevent some future bug 
		// when you start mixing different types of textures.
		// Of course, you still need to correctly set 
		// the image file source format when using gITexImage2D()
		glEnable(GL_FRAMEBUFFER_SRGB);
		glDisable(0x809D);

		glEnable(GL_POLYGON_OFFSET_FILL);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_DEPTH_TEST);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		glFrontFace(GL_CCW);

		glDepthFunc(GL_LEQUAL);

		glDisable(GL_BLEND);
	}

	void GlRendererAPI::SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
	{
		glViewport(x, y, width, height);
	}

	void GlRendererAPI::SetClearColor(const glm::vec4& color)
	{
		glClearColor(color.r, color.g, color.b, color.a);
	}

	void GlRendererAPI::Clear()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void GlRendererAPI::DrawIndexed(const std::shared_ptr<VertexArray>& vertexArray, uint32_t indexCount)
	{
		vertexArray->Bind();
		uint32_t count = indexCount ? indexCount : vertexArray->GetIndexBuffer()->GetCount();
		glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr);
	}

	void GlRendererAPI::DrawLines(const std::shared_ptr<VertexArray>& vertexArray, uint32_t vertexCount)
	{
		vertexArray->Bind();
		glDrawArrays(GL_LINES, 0, vertexCount);
	}

	void GlRendererAPI::SetLineWidth(float width)
	{
		glLineWidth(width);
	}

}