#pragma once
#include <string>

#include "BackInfi/Core/Base.h"

namespace BackInfi
{

	enum class ImageFormat
	{
		NONE = 0,
		R8,
		R32,
		RGB8,
		RGBA8,
		RGBA32F
	};

	struct TextureSpecification
	{
		uint32_t Size      = 4;
		uint32_t Width     = 1;
		uint32_t Height    = 1;
		bool GenerateMips  = true;
		ImageFormat Format = ImageFormat::RGBA8;
	};

	class Texture
	{
	public:
		virtual ~Texture() = default;

		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;
		virtual uint32_t GetRendererID() const = 0;

		virtual std::string GetFilepath() const = 0;

		virtual void LoadTexture(BYTE* data, uint32_t size) const = 0;

		virtual bool IsLoaded() const = 0;

		virtual void Bind(uint32_t slot = 0) const = 0;
		virtual void UnBind(uint32_t slot = 0) const = 0;

		virtual bool operator == (const Texture& texture) const = 0;
	};

	class Texture2D : public Texture
	{
	public:
		static std::shared_ptr<Texture2D> Create(const std::string& filepath);
		static std::shared_ptr<Texture2D> Create(const TextureSpecification& spec);
	};

}