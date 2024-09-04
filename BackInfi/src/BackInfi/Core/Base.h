#pragma once

#ifdef _WIN32
	#ifdef _WIN64
		#define BC_PLATFORM_WINDOWS
	#else
		#error "x86 Builds are not supported!"
	#endif
#elif defined(__APPLE__) || defined(__MACH__)
	#include <TargetConditionals.h>
	/* TARGET_OS_MAC exists on all the platforms
	 * so we must check all of them (in this order)
	 * to ensure that we're running on MAC
	 * and not some other Apple platform */
	#if TARGET_IPHONE_SIMULATOR == 1
		#error "IOS simulator is not supported!"
	#elif TARGET_OS_IPHONE == 1
		#define BC_PLATFORM_IOS
		#error "IOS is not supported!"
	#elif TARGET_OS_MAC == 1
		#define BC_PLATFORM_MACOS
		#error "MacOS is not supported!"
	#else
		#error "Unknown Apple platform!"
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

#define BC_BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }

#include "BackInfi/Core/Assert.h"
#include "BackInfi/Core/Logger.h"