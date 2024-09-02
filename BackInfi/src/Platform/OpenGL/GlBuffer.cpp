#include "bcpch.h"

#include "Platform/OpenGL/GlBuffer.h"

#include <glad/glad.h>

namespace BackInfi
{

	// GlVertexBuffer
	// ---------------------------------------------------------------
	GlVertexBuffer::GlVertexBuffer(uint32_t size)
	{
		glGenBuffers(1, &m_RendererID);
		glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
		glBufferData(
			GL_ARRAY_BUFFER,      // The tanget buffer type
			size,                 // The size in bytes of the buffer object's new data store
			nullptr,              // A pointer to the data that will be copied into the data store
			GL_DYNAMIC_DRAW       // The expected usage pattern of the data store
		);
	}

	GlVertexBuffer::GlVertexBuffer(float* vertices, uint32_t size)
	{
		glGenBuffers(1, &m_RendererID);
		glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
		glBufferData(
			GL_ARRAY_BUFFER,           // The target buffer type
			size,                      // The size in bytes of the buffer object's new data store
			vertices,                  // A pointer to the data that will be copied into the data store
			GL_STATIC_DRAW             // The expected usage pattern of the data store
		);
	}

	GlVertexBuffer::~GlVertexBuffer()
	{
		glDeleteBuffers(1, &m_RendererID);
	}

	void GlVertexBuffer::Bind() const
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
	}

	void GlVertexBuffer::UnBind() const
	{
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void GlVertexBuffer::SetData(const void* data, uint32_t size)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
		glBufferSubData(
			GL_ARRAY_BUFFER,           // The target buffer type
			0,                         // Data offset
			size,                      // The size in bytes of the buffer object's new data store
			data                       // A pointer to the data that will be copied into the data store
		);
	}

	// GlIndexBuffer
	// ---------------------------------------------------------------
	GlIndexBuffer::GlIndexBuffer(uint32_t* indices, uint32_t count)
	{
		m_Count = count;
		glGenBuffers(1, &m_RendererID);

		// GL_ELEMENT_ARRAY_BUFFER is not valid without an actively bound VAO
		// Binding with GL_ARRAY_BUFFER allows the data to be loaded regardless of VAO state. 
		glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
		glBufferData(
			GL_ARRAY_BUFFER,           // The target buffer type
			count * sizeof(uint32_t),  // The size in bytes of the buffer object's new data store
			indices,                   // A pointer to the data that will be copied into the data store
			GL_STATIC_DRAW             // The expected usage pattern of the data store
		);
	}

	GlIndexBuffer::~GlIndexBuffer()
	{
		glDeleteBuffers(1, &m_RendererID);
	}

	void GlIndexBuffer::Bind() const
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID);
	}

	void GlIndexBuffer::UnBind() const
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

}