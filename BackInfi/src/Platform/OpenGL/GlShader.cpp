#include <bcpch.h>
#include <Platform/OpenGL/GlShader.h>

#include <glm/gtc/type_ptr.hpp>

namespace BackInfi
{

	namespace Utils
	{

		static GLenum ShaderTypeFromString(const std::string& type)
		{
			if (type == "vertex")
				return GL_VERTEX_SHADER;
			if (type == "fragment" || type == "pixel")
				return GL_FRAGMENT_SHADER;
			
			BC_CORE_ASSERT(false, "Unknown shader type!");
			return 0;
		}
	}

	GlShader::GlShader(const std::string& filepath)
	{
		auto source = ReadFile(filepath);
		auto [vertexCode, fragmentCode] = PreProcess(source);

		GLuint vertex;
		GLuint fragment;

		// Vertex shader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vertexCode, NULL);
		glCompileShader(vertex);
		CheckCompileErrors(vertex, "VERTEX");

		// Fragment Shader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fragmentCode, NULL);
		glCompileShader(fragment);
		CheckCompileErrors(fragment, "FRAGMENT");

		// Shader Program
		m_RendererID = glCreateProgram();
		glAttachShader(m_RendererID, vertex);
		glAttachShader(m_RendererID, fragment);

		const GLint attr_location[NUM_ATTRIBUTES] = {
			ATTRIB_VERTEX,
			ATTRIB_TEXTURE_POSITION,
		};
		const GLchar* attr_name[NUM_ATTRIBUTES] = {
			"position",
			"texture_coordinate",
		};

		// Attribute location binding must be set before linking.
		for (int i = 0; i < NUM_ATTRIBUTES; i++) {
			glBindAttribLocation(m_RendererID, attr_location[i], attr_name[i]);
		}

		glLinkProgram(m_RendererID);
		CheckCompileErrors(m_RendererID, "PROGRAM");

		// Delete the shaders as they're linked 
		// into our program now and no longer necessary
		glDeleteShader(fragment);
		glDeleteShader(vertex);

		// Extract name from filepath
		auto lastSlash = filepath.find_last_of("/\\");
		lastSlash = lastSlash == std::string::npos ? 0 : lastSlash + 1;
		auto lastDot = filepath.rfind('.');
		auto count = lastDot == std::string::npos ? filepath.size() - lastSlash : lastDot - lastSlash;
		m_Name = filepath.substr(lastSlash, count);
	}

	GlShader::GlShader(
		const std::string& name,
		const std::string& vertexCode,
		const std::string& fragmentCode)
			: m_Name(name)
	{
		const char* vShaderCode = vertexCode.c_str();
		const char* fShaderCode = fragmentCode.c_str();

		GLuint vertex;
		GLuint fragment;

		// Vertex shader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		CheckCompileErrors(vertex, "VERTEX");

		// Fragment Shader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		CheckCompileErrors(fragment, "FRAGMENT");

		// Shader Program
		m_RendererID = glCreateProgram();
		glAttachShader(m_RendererID, vertex);
		glAttachShader(m_RendererID, fragment);

		const GLint attr_location[NUM_ATTRIBUTES] = {
			ATTRIB_VERTEX,
			ATTRIB_TEXTURE_POSITION,
		};
		const GLchar* attr_name[NUM_ATTRIBUTES] = {
			"position",
			"texture_coordinate",
		};

		// Attribute location binding must be set before linking.
		for (int i = 0; i < NUM_ATTRIBUTES; i++) {
			glBindAttribLocation(m_RendererID, attr_location[i], attr_name[i]);
		}

		glLinkProgram(m_RendererID);
		CheckCompileErrors(m_RendererID, "PROGRAM");

		// Delete the shaders as they're linked 
		// into our program now and no longer necessary
		glDeleteShader(fragment);
		glDeleteShader(vertex);
	}

	GlShader::~GlShader()
	{
		GlShader::deleteShader(*this);
	}

	void GlShader::Bind() const
	{
		glUseProgram(m_RendererID);
	}

	void GlShader::UnBind() const
	{
		glUseProgram(0);
	}

	void GlShader::deleteShader(GlShader& shader)
	{
		if (shader.m_RendererID)
			glDeleteProgram(shader.m_RendererID);
		shader.m_RendererID = 0;
		shader.m_Name = "";
	}

	GLint GlShader::GetUniformLocation(const std::string& name) const
	{
		if (m_UmapUniformLocationCache.find(name) != m_UmapUniformLocationCache.end())
			return m_UmapUniformLocationCache[name];

		GLint location = glGetUniformLocation(this->m_RendererID, name.c_str());
		m_UmapUniformLocationCache[name] = location;
		return location;
	}

	void GlShader::CheckCompileErrors(unsigned int shader, std::string type)
	{
		int success;
		char infoLog[1024];
		if (type != "PROGRAM")
		{
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

			if (!success)
			{
				glGetShaderInfoLog(shader, 1024, NULL, infoLog);
				BC_CORE_ERROR("ERROR::SHADER_COMPILATION_ERROR of type: {0} \n {1} \n", type, infoLog);
			}
		}
		else
		{
			glGetProgramiv(shader, GL_LINK_STATUS, &success);

			if (!success)
			{
				glGetProgramInfoLog(shader, 1024, NULL, infoLog);
				BC_CORE_ERROR("ERROR::PROGRAM_LINKING_ERROR of type: {0} \n {1} \n", type, infoLog);
			}
		}
	}

	std::string GlShader::ReadFile(const std::string& filepath)
	{
		std::string result;
		std::ifstream in(filepath, std::ios::in | std::ios::binary); // ifstream closes itself due to RAII
		if (in)
		{
			in.seekg(0, std::ios::end);
			size_t size = in.tellg();
			if (size != -1)
			{
				result.resize(size);
				in.seekg(0, std::ios::beg);
				in.read(&result[0], size);
			}
			else
			{
				BC_CORE_ERROR("Could not read from file '{0}'", filepath);
			}
		}
		else
		{
			BC_CORE_ERROR("Could not open file '{0}'", filepath);
		}

		return result;
	}

	std::tuple<const char*, const char*> GlShader::PreProcess(const std::string& source)
	{
		std::string vertexSrc, fragmentSrc;
		const char* typeToken = "#type";
		size_t typeTokenLength = strlen(typeToken);
		size_t pos = source.find(typeToken, 0); //Start of shader type declaration line
		while (pos != std::string::npos)
		{
			size_t eol = source.find_first_of("\r\n", pos); //End of shader type declaration line
			BC_CORE_ASSERT(eol != std::string::npos, "Syntax error");
			size_t begin = pos + typeTokenLength + 1; //Start of shader type name (after "#type " keyword)
			std::string type = source.substr(begin, eol - begin);
			BC_CORE_ASSERT(Utils::ShaderTypeFromString(type), "Invalid shader type specified");

			size_t nextLinePos = source.find_first_not_of("\r\n", eol); //Start of shader code after shader type declaration line
			BC_CORE_ASSERT(nextLinePos != std::string::npos, "Syntax error");
			pos = source.find(typeToken, nextLinePos); //Start of next shader type declaration line

			if (Utils::ShaderTypeFromString(type) == GL_VERTEX_SHADER)
				vertexSrc = (pos == std::string::npos) ? source.substr(nextLinePos) : source.substr(nextLinePos, pos - nextLinePos);
			else
				fragmentSrc = (pos == std::string::npos) ? source.substr(nextLinePos) : source.substr(nextLinePos, pos - nextLinePos);
		}

		return { vertexSrc.c_str(), fragmentSrc.c_str()};
	}

	void GlShader::SetInt(const std::string& name, const int& value) const
	{
		GLint location = GetUniformLocation(name);
		glUniform1i(location, value);
	}

	void GlShader::SetBool(const std::string& name, const bool& value) const
	{
		GLint location = GetUniformLocation(name);
		glUniform1i(location, static_cast<int>(value));
	}

	void GlShader::SetFloat(const std::string& name, const float& value) const
	{
		GLint location = GetUniformLocation(name);
		glUniform1f(location, value);
	}

	void GlShader::SetVec2f(const std::string& name, const glm::vec2& value) const
	{
		GLint location = GetUniformLocation(name);
		glUniform2f(location, value.x, value.y);
	}

	void GlShader::SetVec3f(const std::string& name, const glm::vec3& value) const
	{
		GLint location = GetUniformLocation(name);
		glUniform3f(location, value.x, value.y, value.z);
	}

	void GlShader::SetVec4f(const std::string& name, const glm::vec4& value) const
	{
		GLint location = GetUniformLocation(name);
		glUniform4f(location, value.x, value.y, value.z, value.w);
	}

	void GlShader::SetIntArray(const std::string& name, int* values, uint32_t count) const
	{
		GLint location = glGetUniformLocation(m_RendererID, name.c_str());
		glUniform1iv(location, count, values);
	}

	void GlShader::SetMat4(const std::string& name, const glm::mat4& matrix) const
	{
		GLint location = glGetUniformLocation(m_RendererID, name.c_str());
		glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
	}

}



