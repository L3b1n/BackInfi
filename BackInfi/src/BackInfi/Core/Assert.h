#pragma once

#include "BackInfi/Core/Base.h"
#include "BackInfi/Core/Logger.h"
#include <filesystem>

#ifdef BC_ENABLE_ASSERTS
	// Alteratively we could use the same "default" message for both "WITH_MSG" and "NO_MSG" and
	// provide support for custom formatting by concatenating the formatting string instead of having the format inside the default message
	#define BC_INTERNAL_ASSERT_IMPL(type, check, msg, ...) { if(!(check)) { BC##type##ERROR(msg, __VA_ARGS__); BC_DEBUGBREAK(); } }
	#define BC_INTERNAL_ASSERT_WITH_MSG(type, check, ...) BC_INTERNAL_ASSERT_IMPL(type, check, "Assertion failed: {0}", __VA_ARGS__)
	#define BC_INTERNAL_ASSERT_NO_MSG(type, check) BC_INTERNAL_ASSERT_IMPL(type, check, "Assertion '{0}' failed at {1}:{2}", BC_STRINGIFY_MACRO(check), std::filesystem::path(__FILE__).filename().string(), __LINE__)

	#define BC_INTERNAL_ASSERT_GET_MACRO_NAME(arg1, arg2, macro, ...) macro
	#define BC_INTERNAL_ASSERT_GET_MACRO(...) BC_EXPAND_MACRO( BC_INTERNAL_ASSERT_GET_MACRO_NAME(__VA_ARGS__, BC_INTERNAL_ASSERT_WITH_MSG, BC_INTERNAL_ASSERT_NO_MSG) )

	// Currently accepts at least the condition and one additional parameter (the message) being optional
	#define BC_ASSERT(...) BC_EXPAND_MACRO( BC_INTERNAL_ASSERT_GET_MACRO(__VA_ARGS__)(_, __VA_ARGS__) )
	#define BC_CORE_ASSERT(...) BC_EXPAND_MACRO( BC_INTERNAL_ASSERT_GET_MACRO(__VA_ARGS__)(_CORE_, __VA_ARGS__) )
#else
	#define BC_ASSERT(...)
	#define BC_CORE_ASSERT(...)
#endif