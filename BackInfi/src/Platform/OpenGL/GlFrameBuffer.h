#pragma once

#include "BackInfi/Renderer/FrameBuffer.h"

namespace BackInfi
{

	class GlFrameBuffer : public FrameBuffer
	{
	public:
		GlFrameBuffer(const FrameBufferSpecs& specs);
		~GlFrameBuffer();

		virtual void Bind() const override;
		virtual void UnBind() const override;

		

		virtual void Resize(uint32_t height, uint32_t width) override;
		virtual BYTE* ReadPixels(uint32_t attachmentIndex, int x, int y) override;

		virtual void ClearAttachment(uint32_t attachmentIndex, int value) override;

		virtual uint32_t GetRendererID() const override { return m_Specs.Samples == 0 ? m_FboID : m_FboMsaaID; }
		virtual uint32_t GetColorAttachmentID(uint32_t index = 0) const override { Update();  return m_TexID; }
		virtual uint32_t GetDepthAttachmentID() const override { return m_RboID; }

		virtual const FrameBufferSpecs& GetSpecs() const override { return m_Specs; }

	private:
		void Invalidate();
		void DeleteFrameBuffer();

		void Update() const;

		void BlitColorTo(uint32_t dstId, int x, int y, int width, int height) const;
		void BlitDepthTo(uint32_t dstId, int x, int y, int width, int height) const;

	private:
		BYTE*    m_ColorBuffer;
		float*   m_DepthBuffer;

		uint32_t m_FboMsaaID;
		uint32_t m_FboID;

		uint32_t m_RboMsaaColorID;
		uint32_t m_RboMsaaDepthID;
			   
		uint32_t m_TexID;
		uint32_t m_RboID;

		FrameBufferSpecs m_Specs;
	};

}