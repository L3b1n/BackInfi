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
#define BC_CORE_TRACE(...)    ::BackInfi::Logger::GetCoreLogger()->trace(__VA_ARGS__)
#define BC_CORE_INFO(...)     ::BackInfi::Logger::GetCoreLogger()->info(__VA_ARGS__)
#define BC_CORE_WARN(...)     ::BackInfi::Logger::GetCoreLogger()->warn(__VA_ARGS__)
#define BC_CORE_ERROR(...)    ::BackInfi::Logger::GetCoreLogger()->error(__VA_ARGS__)
#define BC_CORE_CRITICAL(...) ::BackInfi::Logger::GetCoreLogger()->critical(__VA_ARGS__)