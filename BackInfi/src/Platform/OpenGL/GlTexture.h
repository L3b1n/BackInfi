#pragma once
#include "BackInfi/Renderer/Texture.h"

#include <glad/glad.h>

namespace BackInfi
{
	class GlTexture : public Texture2D
	{
	public:
		GlTexture(const std::string& filepath);
		GlTexture(const TextureSpecification& spec);
		virtual ~GlTexture();

		virtual uint32_t GetWidth() const override { return m_Width; }
		virtual uint32_t GetHeight() const override { return m_Height; }
		virtual uint32_t GetRendererID() const override { return m_RendererID; }

		virtual std::string GetFilepath() const override { return m_Path; }

		virtual void LoadTexture(BYTE* data, uint32_t size) const override;

		virtual bool IsLoaded() const override { return m_IsLoaded; }

		virtual void Bind(uint32_t slot = 0) const override;
		virtual void UnBind(uint32_t slot = 0) const override;

		virtual bool operator == (const Texture& texture) const override
		{
			return m_RendererID == texture.GetRendererID();
		}

	private:
		uint32_t m_Width;
		uint32_t m_Height;

		std::string m_Path;
		bool m_IsLoaded = false;

		TextureSpecification m_Specs;

		GLuint m_RendererID;
		GLenum m_DataFormat;
		GLenum m_InternalFormat;
	};
}