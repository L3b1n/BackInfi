#pragma once

#include <string>
#include <memory>

#include <glm/glm.hpp>

namespace BackInfi
{

	class Shader
	{
	public:
		virtual ~Shader() = default;

		virtual void Bind() const = 0;
		virtual void UnBind() const = 0;

		virtual void SetInt(const std::string& name, const int& value) const = 0;
		virtual void SetBool(const std::string& name,const bool& value) const = 0;
		virtual void SetFloat(const std::string& name, const float& value) const = 0;
		virtual void SetVec2f(const std::string& name, const glm::vec2& value) const = 0;
		virtual void SetVec3f(const std::string& name, const glm::vec3& value) const = 0;
		virtual void SetVec4f(const std::string& name, const glm::vec4& value) const = 0;
		virtual void SetIntArray(const std::string& name, int* values, uint32_t count) const = 0;
		virtual void SetMat4(const std::string& name, const glm::mat4& value) const = 0;

		virtual const std::string& GetName() const = 0;

		static std::shared_ptr<Shader> Create(const std::string& filepath);
		static std::shared_ptr<Shader> Create(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc);
	};

}