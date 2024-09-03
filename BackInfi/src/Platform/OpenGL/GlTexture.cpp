#include "bcpch.h"

#include "Platform/OpenGL/GlTexture.h"

namespace BackInfi
{

	namespace Utils
	{

		static uint32_t GetStep(GLenum format)
		{
			switch (format)
			{
			case GL_RED:
				return 1;
			case GL_RGB:
				return 3;
			case GL_RGBA:
				return 4;
			}

			BC_CORE_ASSERT(false);
			return 0;
		}

		static GLenum ImageFormatToGLDataFormat(ImageFormat format)
		{
			switch (format)
			{
			case ImageFormat::R8:
				return GL_RED;
			case ImageFormat::R32:
				return GL_RED;
			case ImageFormat::RGB8:
				return GL_RGB;
			case ImageFormat::RGBA8:
				return GL_RGBA;
			}

			BC_CORE_ASSERT(false);
			return 0;
		}

		static GLenum ImageFormatToGLInternalFormat(ImageFormat format)
		{
			switch (format)
			{
			case ImageFormat::R8:
				return GL_R8;
			case ImageFormat::R32:
				return GL_R32F;
			case ImageFormat::RGB8:
				return GL_RGB8;
			case ImageFormat::RGBA8:
				return GL_RGBA8;
			}

			BC_CORE_ASSERT(false);
			return 0;
		}

	}

	GlTexture::GlTexture(const std::string& filepath)
	{
		
	}

	GlTexture::GlTexture(const TextureSpecification& spec)
	{
		m_Specs  = spec;
		m_Width  = m_Specs.Width;
		m_Height = m_Specs.Height;

		m_DataFormat     = Utils::ImageFormatToGLDataFormat(m_Specs.Format);
		m_InternalFormat = Utils::ImageFormatToGLInternalFormat(m_Specs.Format);

		// Generate texture
		glGenTextures(1, &m_RendererID);
		glBindTexture(GL_TEXTURE_2D, m_RendererID);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(
			GL_TEXTURE_2D,                                // Type of texture
			0,                                            // Pyramid level (for mip-mapping) - 0 is the top level
			m_InternalFormat,                             // Internal colour format to convert to
			m_Width,                                      // Image width i.e. width member of BackgroundFilter in standard mode
			m_Height,                                     // Image height i.e. height member of BackgroundFilter in standard mode
			0,                                            // Border width in pixels (can either be 1 or 0)
			m_DataFormat,                                 // Input image format (i.e. GL_RED, GL_RGB, GL_RGBA, GL_BGR etc.)
			m_DataFormat 
				== GL_RED ? GL_FLOAT : GL_UNSIGNED_BYTE,  // Image data type
			nullptr                                       // Here I provide nullptr, because I just want to create texture. Data binds in LoadTexture
		);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	GlTexture::~GlTexture()
	{
		if (m_RendererID)
		{
			glDeleteTextures(1, &m_RendererID);
			m_RendererID = 0;
		}
	}

	void GlTexture::LoadTexture(BYTE* data, uint32_t size) const
	{
		uint32_t bbp = Utils::GetStep(m_DataFormat);
		BC_CORE_ASSERT(size == m_Width * m_Height * bbp, "Data must be entire texture!");
		glBindTexture(GL_TEXTURE_2D, m_RendererID);
		glTexSubImage2D(
			GL_TEXTURE_2D,                             // Type of texture
			0,                                         // Pyramid level (for mip-mapping) - 0 is the top level
			0,                                         // Specifies a texel offset in the x direction within the texture array
			0,                                         // Specifies a texel offset in the y direction within the texture array
			m_Width,                                   // Image width
			m_Height,                                  // Image height
			m_DataFormat,                              // Input image format (i.e. GL_RED, GL_RGB, GL_RGBA, GL_BGR etc.)
			m_DataFormat == 
				GL_RED ? GL_FLOAT : GL_UNSIGNED_BYTE,  // Image data type
			data                                       // The actual image data itself
		);
		// Generate mipmaps for color buffer (texture)
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void GlTexture::Bind(uint32_t slot) const
	{
		glActiveTexture(GL_TEXTURE0 + slot);
		glBindTexture(GL_TEXTURE_2D, m_RendererID);
	}

	void GlTexture::UnBind(uint32_t slot) const
	{
		glActiveTexture(GL_TEXTURE0 + slot);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

}