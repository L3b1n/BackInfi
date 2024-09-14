-- Include directories relative to root folder (solution directory)
IncludeDir = {}
IncludeDir["glm"]         = "vendor/glm"
IncludeDir["GLFW"]        = "vendor/GLFW/include"
IncludeDir["Glad"]        = "vendor/Glad/include"
IncludeDir["ImGui"]       = "vendor/imgui"
IncludeDir["Opencv"]      = "vendor/Opencv/include"
IncludeDir["Onnxruntime"] = "vendor/Onnxruntime/include"

LibsDir = {}
LibsDir["Opencv"]         = "vendor/Opencv/Lib"
LibsDir["Onnxruntime"]    = "vendor/Onnxruntime/Lib"

include "vendor/GLFW"
include "vendor/Glad"
include "vendor/ImGui"

project "BackInfi"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

	pchheader "bcpch.h"
	pchsource "src/bcpch.cpp"

	files
	{
		"src/**.h",
		"src/**.cpp",
		"Sandbox/**.h",
		"Sandbox/**.cpp",
		"vendor/glm/glm/**.hpp",
		"vendor/glm/glm/**.inl"
	}

	defines
	{
		"_CRT_SECURE_NO_WARNINGS",
		"GLFW_INCLUDE_NONE"
	}

	includedirs
	{
		"src",
		"Sandbox",
		"vendor/spdlog/include",
		"%{IncludeDir.glm}",
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.ImGui}",
		"%{IncludeDir.Opencv}",
		"%{IncludeDir.Onnxruntime}"
	}

	libdirs
	{
		"%{LibsDir.Opencv}",
		"%{LibsDir.Onnxruntime}"
	}

	links
	{
		"GLFW",
		"Glad",
		"ImGui",
		"gdi32.lib",
		"Winmm.lib",
		"Version.lib",
		"opengl32.lib",
	}

	filter "system:windows"
		systemversion "latest"

		defines
		{
		}

	filter "configurations:Debug"
		defines
		{
			"BC_DEBUG",
			"BC_PROFILE"
		}
		runtime "Debug"
		symbols "on"

		links
		{
			"aded.lib",
			"IlmImfd.lib",
			"ippiwd.lib",
			"ittnotifyd.lib",
			"libjpeg-turbod.lib",
			"libopenjp2d.lib",
			"libpngd.lib",
			"libprotobufd.lib",
			"libtiffd.lib",
			"libwebpd.lib",
			"opencv_world480d.lib",
			"quircd.lib",
			"zlibd.lib",
			"onnxruntime_d.lib"
		}

	filter "configurations:Release"
		defines
		{
			"BC_RELEASE",
			"BC_PROFILE"
		}
		runtime "Release"
		optimize "on"

		links
		{
			"ade.lib",
			"IlmImf.lib",
			"ippicvmt.lib",
			"ippiw.lib",
			"ittnotify.lib",
			"libjpeg-turbo.lib",
			"libopenjp2.lib",
			"libpng.lib",
			"libprotobuf.lib",
			"libtiff.lib",
			"libwebp.lib",
			"opencv_world480.lib",
			"quirc.lib",
			"zlib.lib",
			"onnxruntime.lib"
		}

	filter "configurations:Dist"
		defines "BC_DIST"
		runtime "Release"
		optimize "on"

		links
		{
			"ade.lib",
			"IlmImf.lib",
			"ippicvmt.lib",
			"ippiw.lib",
			"ittnotify.lib",
			"libjpeg-turbo.lib",
			"libopenjp2.lib",
			"libpng.lib",
			"libprotobuf.lib",
			"libtiff.lib",
			"libwebp.lib",
			"opencv_world480.lib",
			"quirc.lib",
			"zlib.lib",
			"onnxruntime.lib"
		}
