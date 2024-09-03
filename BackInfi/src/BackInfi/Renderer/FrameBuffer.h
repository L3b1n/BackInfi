#pragma once

#include "BackInfi/Core/Base.h"

namespace BackInfi
{

	enum class TexFormat
	{
		None = 0,

		// Color attachment
		RGB8,
		RGBA8,
		RED_INTEGER,

		// Depth/stencil
		DEPTH24STENCIL8,

		// Defaults
		Depth = DEPTH24STENCIL8
	};

	enum class TexFilterFormat
	{
		None = 0,

		Nearest,
		NearestMipMapNearest,
		NearestMipMapLinear,

		Linear,
		LinearMipMapLinear,
		LinearMipMapNearest,
	};

	enum class TexWrapFormat
	{
		None = 0,

		Repeat,
		ClampToEdge,
		ClampToBorder,
		MirroredRepeat,
		MirrorClampToEdge
	};

	struct TexFilter
	{
		TexFilter() = default;
		TexFilter(TexFilterFormat mag, TexFilterFormat min)
		{
			Mag = mag;
			Min = min;
		}

		TexFilterFormat Mag;
		TexFilterFormat Min;
	};

	struct TexWrap
	{
		TexWrap() = default;
		TexWrap(TexWrapFormat s, TexWrapFormat t)
		{
			S = s;
			T = t;
		}

		TexWrapFormat S;
		TexWrapFormat T;
	};

	struct FrameBufferSpecs
	{
		uint32_t Size    = 0;
		uint32_t Width   = 0;
		uint32_t Height  = 0;
		uint32_t Samples = 0;

		TexWrap   TextureWrap;
		TexFilter TextureFilter;
		TexFormat TextureFormat = TexFormat::None;

		bool SwapChainTarget = false;
	};

	class FrameBuffer
	{
	public:
		virtual ~FrameBuffer() = default;

		virtual void Bind() const = 0;
		virtual void UnBind() const = 0;

		virtual void Resize(uint32_t height, uint32_t width) = 0;
		virtual BYTE* ReadPixels(uint32_t attachmentIndex, int x, int y) = 0;

		virtual void ClearAttachment(uint32_t attachmentIndex, int value) = 0;

		virtual uint32_t GetRendererID() const = 0;
		virtual uint32_t GetColorAttachmentID(uint32_t index = 0) const = 0;
		virtual uint32_t GetDepthAttachmentID() const = 0;

		virtual const FrameBufferSpecs& GetSpecs() const = 0;

		static std::shared_ptr<FrameBuffer> Create(const FrameBufferSpecs& specs);
	};

}
