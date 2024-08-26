#pragma once

#ifdef _WIN32
	#ifdef _WIN64
		#define BC_PLATFORM_WINDOWS
	#else
		#error "x86 Builds are not supported!"
	#endif
#elif defined(__linux__)
	#define BC_PLATFORM_LINUX
	#error "Linux is not supported!"
#else

	#error "Unknown platform!"
#endif

#ifdef BC_DEBUG
	#if defined(BC_PLATFORM_WINDOWS)
		#define BC_DEBUGBREAK() __debugbreak()
	#elif defined(BC_PLATFORM_LINUX)
		#include <signal.h>
		#define BC_DEBUGBREAK() raise(SIGTRAP)
	#else
		#error "Platform doesn't support debugbreak yet!"
	#endif
	#define BC_ENABLE_ASSERTS
#else
	#define BC_DEBUGBREAK()
#endif

#define BC_EXPAND_MACRO(x) x
#define BC_STRINGIFY_MACRO(x) #x

#define BIT(x) (1 << x)