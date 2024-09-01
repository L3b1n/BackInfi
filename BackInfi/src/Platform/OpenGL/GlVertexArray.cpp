#include "bcpch.h"

#include "Platform/OpenGL/GlVertexArray.h"

#include <glad/glad.h>

namespace BackInfi
{
	namespace Utils
	{

		static GLenum ShaderDataTypeToOpenGLBaseType(ShaderDataType type)
		{
			switch (type)
			{
			case ShaderDataType::Float:    return GL_FLOAT;
			case ShaderDataType::Float2:   return GL_FLOAT;
			case ShaderDataType::Float3:   return GL_FLOAT;
			case ShaderDataType::Float4:   return GL_FLOAT;
			case ShaderDataType::Mat3:     return GL_FLOAT;
			case ShaderDataType::Mat4:     return GL_FLOAT;
			case ShaderDataType::Int:      return GL_INT;
			case ShaderDataType::Int2:     return GL_INT;
			case ShaderDataType::Int3:     return GL_INT;
			case ShaderDataType::Int4:     return GL_INT;
			case ShaderDataType::Bool:     return GL_BOOL;
			}

			BC_CORE_ASSERT(false, "Unknown ShaderDataType!");
			return 0;
		}

	}

	GlVertexArray::GlVertexArray()
	{
		//glCreateVertexArrays(1, &m_RendererID);
		glGenVertexArrays(1, &m_RendererID);
		glBindVertexArray(m_RendererID);
	}

	GlVertexArray::~GlVertexArray()
	{
		glDeleteVertexArrays(1, &m_RendererID);
	}

	void GlVertexArray::Bind() const
	{
		glBindVertexArray(m_RendererID);
	}

	void GlVertexArray::UnBind() const
	{
		glBindVertexArray(0);
	}

	void GlVertexArray::AddVertexBuffer(const std::shared_ptr<VertexBuffer>& vertexBuffer)
	{
		BC_CORE_ASSERT(vertexBuffer->GetLayout().GetElements().size(), "Vertex Buffer has no layout!");

		glBindVertexArray(m_RendererID);
		vertexBuffer->Bind();

		const auto& layout = vertexBuffer->GetLayout();
		for (const auto& element : layout)
		{
			switch (element.Type)
			{
			case ShaderDataType::Float:
			case ShaderDataType::Float2:
			case ShaderDataType::Float3:
			case ShaderDataType::Float4:
			{
				glEnableVertexAttribArray(m_VertexBufferIndex);
				glVertexAttribPointer(
					m_VertexBufferIndex,                                            // Attribute 0 -- The layout position in the shader
					element.GetComponentCount(),                                    // Size        -- Number of components per vertex
					Utils::ShaderDataTypeToOpenGLBaseType(element.Type),            // Type        -- The data type of the above components
					element.Normalized ? GL_TRUE : GL_FALSE,                        // Normalized  -- Specifies if fixed - point data values should be normalized
					layout.GetStride(),                                             // Stride      -- Specifies the byte offset between consecutive attributes
					(const void*)element.Offset                                     // Pointer     -- Specifies the offset of the first component
				);
				m_VertexBufferIndex++;
				break;                
			}
			case ShaderDataType::Int:
			case ShaderDataType::Int2:
			case ShaderDataType::Int3:
			case ShaderDataType::Int4:
			case ShaderDataType::Bool:
			{
				glEnableVertexAttribArray(m_VertexBufferIndex);
				glVertexAttribIPointer(
					m_VertexBufferIndex,                                            // Attribute 0 -- The layout position in the shader  
					element.GetComponentCount(),                                    // Size        -- Number of components per vertex
					Utils::ShaderDataTypeToOpenGLBaseType(element.Type),            // Type        -- The data type of the above components
					layout.GetStride(),                                             // Stride      -- Specifies the byte offset between consecutive attributes
					(const void*)element.Offset                                     // Pointer     -- Specifies the offset of the first component
				);
				m_VertexBufferIndex++;
				break;
			}
			case ShaderDataType::Mat3:
			case ShaderDataType::Mat4:
			{
				uint8_t count = element.GetComponentCount();
				for (uint8_t i = 0; i < count; i++)
				{
					glEnableVertexAttribArray(m_VertexBufferIndex);
					glVertexAttribPointer(
						m_VertexBufferIndex,                                        // Attribute 0 -- The layout position in the shader
						count,                                                      // Size        -- Number of components per vertex
						Utils::ShaderDataTypeToOpenGLBaseType(element.Type),        // Type        -- The data type of the above components
						element.Normalized ? GL_TRUE : GL_FALSE,                    // Normalized  -- Specifies if fixed - point data values should be normalized
						layout.GetStride(),                                         // Stride      -- Specifies the byte offset between consecutive attributes
						(const void*)(element.Offset + sizeof(float) * count * i)   // Pointer     -- Specifies the offset of the first component
					);
					glVertexAttribDivisor(m_VertexBufferIndex, 1);
					m_VertexBufferIndex++;
				}
				break;
			}
			default:
				BC_CORE_ASSERT(false, "Unknown ShaderDataType!");
			}
		}

		m_VertexBuffers.push_back(vertexBuffer);
	}

	void GlVertexArray::SetIndexBuffer(const std::shared_ptr<IndexBuffer>& indexBuffer)
	{
		glBindVertexArray(m_RendererID);
		indexBuffer->Bind();

		m_IndexBuffer = indexBuffer;
	}

}
