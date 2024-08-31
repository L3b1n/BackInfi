#include "bcpch.h"

#include "BackInfi/Core/Window.h"

#ifdef BC_PLATFORM_WINDOWS
	#include "Platform/Windows/WindowsWindow.h"
#endif

namespace BackInfi
{

	std::unique_ptr<Window> Window::Create(const WindowProp& props)
	{
	#ifdef BC_PLATFORM_WINDOWS
		return std::make_unique<WindowsWindow>(props);
	#else
		BC_CORE_ASSERT(false, "Unknown platform!");
		return nullptr;
	#endif
	}

}