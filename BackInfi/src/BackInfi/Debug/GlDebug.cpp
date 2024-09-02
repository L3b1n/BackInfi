#include "bcpch.h"

#include "BackInfi/Debug/GlDebug.h"

namespace BackInfi::Debug
{

	static DebugLogLevel s_DebugLogLevel = DebugLogLevel::HighAssert;

	void SetGLDebugLogLevel(DebugLogLevel level)
	{
		s_DebugLogLevel = level;
	}

	void OpenGLLogMessage(
		GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const GLchar* message,
		const void* userParam)
	{
		switch (severity)
		{
		case GL_DEBUG_SEVERITY_HIGH: {
			if ((int)s_DebugLogLevel > 0)
			{
				BC_CORE_ERROR("[OpenGL Debug HIGH] {0}", message);
				if (s_DebugLogLevel == DebugLogLevel::HighAssert)
					BC_CORE_ASSERT(false, "GL_DEBUG_SEVERITY_HIGH");
			}
			break;
		}
		case GL_DEBUG_SEVERITY_MEDIUM:
			if ((int)s_DebugLogLevel > 2)
				BC_CORE_WARN("[OpenGL Debug MEDIUM] {0}", message);
			break;
		case GL_DEBUG_SEVERITY_LOW:
			if ((int)s_DebugLogLevel > 3)
				BC_CORE_INFO("[OpenGL Debug LOW] {0}", message);
			break;
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			if ((int)s_DebugLogLevel > 4)
				BC_CORE_TRACE("[OpenGL Debug NOTIFICATION] {0}", message);
			break;
		}
	}

	void EnableGLDebugging()
	{
		if (GLAD_GL_VERSION_4_3)
		{
			glDebugMessageCallback(OpenGLLogMessage, nullptr);
			glEnable(GL_DEBUG_OUTPUT);
			glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		}
		else if (GLAD_GL_ARB_debug_output != 0)
		{
			glDebugMessageCallbackARB(OpenGLLogMessage, NULL);
		}
		else
		{
			BC_CORE_WARN("Failed to set GL debug callback as it is "
				"not supported.");
		}
		
	}

}
