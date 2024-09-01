#pragma once

#include "BackInfi/Renderer/Buffer.h"

namespace BackInfi
{

	class GlVertexBuffer : public VertexBuffer
	{
	public:
		GlVertexBuffer(uint32_t size);
		GlVertexBuffer(float* vertices, uint32_t size);

		virtual ~GlVertexBuffer();

		virtual void Bind() const override;
		virtual void UnBind() const override;

		virtual void SetData(const void* data, uint32_t size) override;

		virtual const BufferLayout& GetLayout() const override { return m_Layout; };
		virtual void SetLayout(const BufferLayout& layout) override { m_Layout = layout; }

	private:
		uint32_t m_RendererID;
		BufferLayout m_Layout;
	};

	class GlIndexBuffer : public IndexBuffer
	{
	public:
		GlIndexBuffer(uint32_t* indices, uint32_t count);

		virtual ~GlIndexBuffer();

		virtual void Bind() const override;
		virtual void UnBind() const override;

		virtual uint32_t GetCount() const override { return m_Count; }

	private:
		uint32_t m_RendererID;
		uint32_t m_Count;
	};

}