#pragma once

// This ignores all warnings raised inside External headers
#pragma warning(push, 0)
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#pragma warning(pop)

namespace BackInfi
{
	class Logger
	{
	public:
		static void Init();

		static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_CoreLogger; }
	private:
		static std::shared_ptr<spdlog::logger> s_CoreLogger;
	};
}

// Core log macros
#define LOG_TRACE(...)    ::BackInfi::Logger::GetCoreLogger()->trace(__VA_ARGS__)
#define LOG_INFO(...)     ::BackInfi::Logger::GetCoreLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)     ::BackInfi::Logger::GetCoreLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)    ::BackInfi::Logger::GetCoreLogger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...) ::BackInfi::Logger::GetCoreLogger()->critical(__VA_ARGS__)