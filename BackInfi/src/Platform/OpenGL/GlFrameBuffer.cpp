#include "bcpch.h"

#include "Platform/OpenGL/GlFrameBuffer.h"

#include <glad/glad.h>

namespace BackInfi
{

	static const uint32_t s_MaxFramebufferSize = 8192;

	namespace Utils
	{
		static GLenum TexFormatToGlTexFormat(const TexFormat& format, bool internal)
		{
			switch (format)
			{
			case TexFormat::None:
				BC_CORE_ASSERT(false, "TexFormat::NONE is currently not supported!"); return 0;
			case TexFormat::RGB8:
				return internal ? GL_RGB8 : GL_RGB;
			case TexFormat::RGBA8:
				return internal ? GL_RGBA8 : GL_RGBA;
			case TexFormat::RED_INTEGER:
				return internal ? GL_R32I : GL_RED_INTEGER;
			}

			BC_CORE_ASSERT(false, "Unsupported TexFormat was supplied!");
			return 0;
		}

		static GLenum TexFilterToGlTexFilter(const TexFilterFormat& format)
		{
			switch (format)
			{
			case TexFilterFormat::None:
				BC_CORE_ASSERT(false, "TexFilterFormat::NONE is currently not supported!"); return 0;
			case TexFilterFormat::Nearest:
				return GL_NEAREST;
			case TexFilterFormat::NearestMipMapNearest:
				return GL_NEAREST_MIPMAP_NEAREST;
			case TexFilterFormat::NearestMipMapLinear:
				return GL_NEAREST_MIPMAP_LINEAR;
			case TexFilterFormat::Linear:
				return GL_LINEAR;
			case TexFilterFormat::LinearMipMapLinear:
				return GL_LINEAR_MIPMAP_LINEAR;
			case TexFilterFormat::LinearMipMapNearest:
				return GL_LINEAR_MIPMAP_NEAREST;
			}

			BC_CORE_ASSERT(false, "Unsupported TexFilterFormat was supplied!");
			return 0;
		}

		static GLenum TexWrapToGlTexWrap(const TexWrapFormat& format)
		{
			switch (format)
			{
			case TexWrapFormat::None:
				BC_CORE_ASSERT(false, "TexWrapFormat::NONE is currently not supported!"); return 0;
			case TexWrapFormat::Repeat:
				return GL_REPEAT;
			case TexWrapFormat::ClampToEdge:
				return GL_CLAMP_TO_EDGE;
			case TexWrapFormat::ClampToBorder:
				return GL_CLAMP_TO_BORDER;
			case TexWrapFormat::MirroredRepeat:
				return GL_MIRRORED_REPEAT;
			case TexWrapFormat::MirrorClampToEdge:
				return GL_MIRROR_CLAMP_TO_EDGE;
			}

			BC_CORE_ASSERT(false, "Unsupported TexWrapFormat was supplied!");
			return 0;
		}

		static void CreateTexture(int count, uint32_t* outID)
		{
			glGenTextures(count, outID);
		}

		static void BindTexture(uint32_t id)
		{
			glBindTexture(GL_TEXTURE_2D, id);
		}

		static void AttachColorToTexture(const FrameBufferSpecs& specs, int index, uint32_t id)
		{
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, Utils::TexFilterToGlTexFilter(specs.TextureFilter.Mag));
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, Utils::TexFilterToGlTexFilter(specs.TextureFilter.Min));
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, Utils::TexWrapToGlTexWrap(specs.TextureWrap.S));
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, Utils::TexWrapToGlTexWrap(specs.TextureWrap.T));

			glTexImage2D(
				GL_TEXTURE_2D,
				0,
				Utils::TexFormatToGlTexFormat(specs.TextureFormat, true),
				specs.Width,
				specs.Height,
				0,
				Utils::TexFormatToGlTexFormat(specs.TextureFormat, false),
				GL_UNSIGNED_BYTE,
				nullptr
			);

			glFramebufferTexture2D(
				GL_FRAMEBUFFER,
				GL_COLOR_ATTACHMENT0 + index,
				GL_TEXTURE_2D,
				id,
				0
			);
		}

		static void CheckFrameBufferStatus()
		{
			// Check FBO status
			GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			switch (status)
			{
			case GL_FRAMEBUFFER_COMPLETE:
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
				BC_CORE_ASSERT(false, "[ERROR] Framebuffer incomplete: Attachment is NOT complete.");
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
				BC_CORE_ASSERT(false, "[ERROR] Framebuffer incomplete: No image is attached to FBO.");
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
				BC_CORE_ASSERT(false, "[ERROR] Framebuffer incomplete: Draw buffer.");
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
				BC_CORE_ASSERT(false, "[ERROR] Framebuffer incomplete: Read buffer.");
			case GL_FRAMEBUFFER_UNSUPPORTED:
				BC_CORE_ASSERT(false, "[ERROR] Framebuffer incomplete: Unsupported by FBO implementation.");
			default:
				BC_CORE_ASSERT(false, "[ERROR] Framebuffer incomplete: Unknown error.");
			}
		}

	}

	GlFrameBuffer::GlFrameBuffer(const FrameBufferSpecs& specs)
	{
		BC_PROFILE_FUNC();

		m_Specs          = specs;
		m_ColorBuffer    = nullptr;
		m_DepthBuffer    = nullptr;
		m_FboMsaaID      = 0;
		m_FboID          = 0;
		m_RboMsaaColorID = 0;
		m_RboMsaaDepthID = 0;
		m_TexID          = 0;
		m_RboID          = 0;

		Invalidate();
	}

	GlFrameBuffer::~GlFrameBuffer()
	{
		BC_PROFILE_FUNC();

		GlFrameBuffer::DeleteFrameBuffer();
	}

	void GlFrameBuffer::DeleteFrameBuffer()
	{
		BC_PROFILE_FUNC();

		if (m_RboMsaaColorID)
		{
			glDeleteRenderbuffers(1, &m_RboMsaaColorID);
			m_RboMsaaColorID = 0;
		}
		if (m_RboMsaaDepthID)
		{
			glDeleteRenderbuffers(1, &m_RboMsaaDepthID);
			m_RboMsaaDepthID = 0;
		}
		if (m_FboMsaaID)
		{
			glDeleteFramebuffers(1, &m_FboMsaaID);
			m_FboMsaaID = 0;
		}
		if (m_TexID)
		{
			glDeleteTextures(1, &m_TexID);
			m_TexID = 0;
		}
		if (m_RboID)
		{
			glDeleteRenderbuffers(1, &m_RboID);
			m_RboID = 0;
		}
		if (m_FboID)
		{
			glDeleteFramebuffers(1, &m_FboID);
			m_FboID = 0;
		}
		if (m_ColorBuffer)
		{
			delete[] m_ColorBuffer;
			m_ColorBuffer = 0;
		}
		if (m_DepthBuffer)
		{
			delete[] m_DepthBuffer;
			m_DepthBuffer = 0;
		}
	}

	void GlFrameBuffer::Invalidate()
	{
		BC_PROFILE_FUNC();

		GlFrameBuffer::DeleteFrameBuffer();

		// Check width and height
		BC_CORE_ASSERT(m_Specs.Width > 0 && m_Specs.Height > 0, "The buffer size is not positive.");

		// Validate multi sample count
		int maxMsaa = 0;
		glGetIntegerv(GL_MAX_SAMPLES, &maxMsaa);
		if (m_Specs.Samples < 0)
			m_Specs.Samples = 0;
		else if (m_Specs.Samples > maxMsaa)
			m_Specs.Samples = maxMsaa;
		else if (m_Specs.Samples % 2 != 0)
			m_Specs.Samples--;

		// Create arrays
		m_ColorBuffer = new BYTE[m_Specs.Size];                    // 24 bits per pixel
		m_DepthBuffer = new float[m_Specs.Width * m_Specs.Height]; // 24 bits per pixel

		// Create single-sample FBO
		glGenFramebuffers(1, &m_FboID);
		glBindFramebuffer(GL_FRAMEBUFFER, m_FboID);

		// Create a texture object to store colour info, and attach it to FBO
		Utils::CreateTexture(1, &m_TexID);
		Utils::BindTexture(m_TexID);
		Utils::AttachColorToTexture(m_Specs, 0, m_TexID);

		// Create a renderbuffer object to store depth info, attach it to FBO
		glGenRenderbuffers(1, &m_RboID);
		glBindRenderbuffer(GL_RENDERBUFFER, m_RboID);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_Specs.Width, m_Specs.Height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_RboID);

		// Check FBO completeness
		Utils::CheckFrameBufferStatus();

		// Create multi-sample FBO
		if (m_Specs.Samples > 0)
		{
			glGenFramebuffers(1, &m_FboMsaaID);
			glBindFramebuffer(GL_FRAMEBUFFER, m_FboMsaaID);

			// Create a render buffer object to store colour info
			glGenRenderbuffers(1, &m_RboMsaaColorID);
			glBindRenderbuffer(GL_RENDERBUFFER, m_RboMsaaColorID);
			glRenderbufferStorageMultisample(
				GL_RENDERBUFFER,
				m_Specs.Samples,
				Utils::TexFormatToGlTexFormat(m_Specs.TextureFormat, true),
				m_Specs.Width,
				m_Specs.Height
			);

			// Attach a renderbuffer to FBO color attachment point
			glFramebufferRenderbuffer(
				GL_FRAMEBUFFER,
				GL_COLOR_ATTACHMENT0,
				GL_RENDERBUFFER,
				m_RboMsaaColorID
			);

			// Create a renderbuffer object to store depth info
			glGenRenderbuffers(1, &m_RboMsaaDepthID);
			glBindRenderbuffer(GL_RENDERBUFFER, m_RboMsaaDepthID);
			glRenderbufferStorageMultisample(
				GL_RENDERBUFFER,
				m_Specs.Samples,
				GL_DEPTH_COMPONENT,
				m_Specs.Width,
				m_Specs.Height
			);

			// Attach a renderbuffer to FBO depth attachment point
			glFramebufferRenderbuffer(
				GL_FRAMEBUFFER,
				GL_DEPTH_ATTACHMENT,
				GL_RENDERBUFFER,
				m_RboMsaaDepthID
			);

			// Check FBO completeness again
			Utils::CheckFrameBufferStatus();
		}

		// Unbind
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void GlFrameBuffer::Bind() const
	{
		BC_PROFILE_FUNC();

		glBindFramebuffer(GL_FRAMEBUFFER, m_Specs.Samples == 0 ? m_FboID : m_FboMsaaID);
		glViewport(0, 0, m_Specs.Width, m_Specs.Height);
	}

	void GlFrameBuffer::UnBind() const
	{
		BC_PROFILE_FUNC();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void GlFrameBuffer::Update() const
	{
		BC_PROFILE_FUNC();

		if (m_Specs.Samples > 0)
		{
			// Blit color buffer
			glBindFramebuffer(GL_READ_FRAMEBUFFER, m_FboMsaaID);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_FboID);
			glBlitFramebuffer(
				0, 0, m_Specs.Width, m_Specs.Height,
				0, 0, m_Specs.Width, m_Specs.Height,
				GL_COLOR_BUFFER_BIT,
				GL_LINEAR
			);

			//NOTE: blit separately depth buffer because different scale filter
			//NOTE: scale filter for depth buffer must be GL_NEAREST, otherwise, invalid op
			glBlitFramebuffer(
				0, 0, m_Specs.Width, m_Specs.Height,
				0, 0, m_Specs.Width, m_Specs.Height,
				GL_DEPTH_BUFFER_BIT,
				GL_NEAREST
			);
		}

		// Also, generate mipmaps for color buffer (texture)
		glBindTexture(GL_TEXTURE_2D, m_TexID);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void GlFrameBuffer::BlitColorTo(
		uint32_t dstId,
		int x, int y,
		int width, int height) const
	{
		BC_PROFILE_FUNC();

		// If width/height not specified, use src dimension
		if (width == 0) width = m_Specs.Width;
		if (height == 0) height = m_Specs.Height;

		uint32_t srcId = (m_Specs.Samples == 0) ? m_FboID : m_FboMsaaID;
		glBindFramebuffer(GL_READ_FRAMEBUFFER, srcId);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dstId);
		glBlitFramebuffer(
			0, 0, m_Specs.Width, m_Specs.Height,   // src rect
			x, y, width, height,                   // dst rect
			GL_COLOR_BUFFER_BIT,                   // buffer mask
			GL_LINEAR                              // scale filter
		);
	}

	void GlFrameBuffer::BlitDepthTo(
		uint32_t dstId,
		int x, int y,
		int width, int height) const
	{
		BC_PROFILE_FUNC();

		// If width/height not specified, use src dimension
		if (width == 0) width = m_Specs.Width;
		if (height == 0) height = m_Specs.Height;

		// NOTE: scale filter for depth buffer must be GL_NEAREST, otherwise, invalid op
		uint32_t srcId = (m_Specs.Samples == 0) ? m_FboID : m_FboMsaaID;
		glBindFramebuffer(GL_READ_FRAMEBUFFER, srcId);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dstId);

		glBlitFramebuffer(
			0, 0, m_Specs.Width, m_Specs.Height,   // src rect
			x, y, width, height,                   // dst rect
			GL_DEPTH_BUFFER_BIT,                   // buffer mask
			GL_NEAREST                             // scale filter
		);
	}

	void GlFrameBuffer::Resize(uint32_t height, uint32_t width)
	{
		BC_PROFILE_FUNC();

		if (width == 0 || height == 0 || width > s_MaxFramebufferSize || height > s_MaxFramebufferSize)
		{
			BC_CORE_WARN("Attempted to rezize framebuffer to {0}, {1}", width, height);
			return;
		}
		m_Specs.Width  = width;
		m_Specs.Height = height;

		Invalidate();
	}

	BYTE* GlFrameBuffer::ReadPixels(uint32_t attachmentIndex, int x, int y)
	{
		BC_PROFILE_FUNC();

		//Update();
		if (m_Specs.Samples > 0)
		{
			// Copy multi-sample to single-sample first
			BlitColorTo(m_FboID, 0, 0, m_Specs.Width, m_Specs.Height);
		}
		// Store pixel data to internal array
		glBindFramebuffer(GL_READ_FRAMEBUFFER, m_FboID);

		// Set to GL_RGB due to shader output format I want
		glReadPixels(
			x,
			y,
			m_Specs.Width,
			m_Specs.Height,
			Utils::TexFormatToGlTexFormat(m_Specs.TextureFormat, false),
			GL_UNSIGNED_BYTE,
			m_ColorBuffer
		);
		
		return m_ColorBuffer;
	}

	void GlFrameBuffer::ClearAttachment(uint32_t attachmentIndex, int value)
	{
	}

}
