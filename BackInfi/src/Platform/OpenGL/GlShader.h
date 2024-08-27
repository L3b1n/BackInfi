#pragma once
#include <string>
#include <unordered_map>

#include <glm/glm.hpp>
#include <glad/glad.h>

#include <BackInfi/Renderer/Shader.h>

namespace BackInfi
{

	enum
	{ 
		ATTRIB_VERTEX,
		ATTRIB_TEXTURE_POSITION,
		NUM_ATTRIBUTES
	};

	class GlShader : public Shader
	{
	public:
		GlShader(const std::string& filepath);
		GlShader(
			const std::string& name,
			const std::string& vertexCode,
			const std::string& fragmentCode);
		virtual ~GlShader();

		virtual void Bind() const override;
		virtual void UnBind() const override;

		virtual void SetInt(const std::string& name, const int& value) const override;
		virtual void SetBool(const std::string& name, const bool& value) const override;
		virtual void SetFloat(const std::string& name, const float& value) const override;
		virtual void SetVec2f(const std::string& name, const glm::vec2& value) const override;
		virtual void SetVec3f(const std::string& name, const glm::vec3& value) const override;
		virtual void SetVec4f(const std::string& name, const glm::vec4& value) const override;
		virtual void SetIntArray(const std::string& name, int* values, uint32_t count) const override;
		virtual void SetMat4(const std::string& name, const glm::mat4& value) const override;

		virtual const std::string& GetName() const override { return m_Name; }

	private:
		static void deleteShader(GlShader& shader);
		GLint GetUniformLocation(const std::string& name) const;
		void CheckCompileErrors(unsigned int shader, std::string type);

		std::string ReadFile(const std::string& filepath);
		std::tuple<const char*, const char*> PreProcess(const std::string& source);

	private:
		std::string m_Name;
		GLuint m_RendererID; // Program ID
		mutable std::unordered_map<std::string, GLint> m_UmapUniformLocationCache;
	};

}

