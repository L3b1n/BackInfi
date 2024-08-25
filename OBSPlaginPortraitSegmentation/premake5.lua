-- Include directories relative to root folder (solution directory)
IncludeDir = {}
IncludeDir["GLFW"]        = "vendor/GLFW/include"
IncludeDir["Glad"]        = "vendor/Glad/include"
IncludeDir["Opencv"]      = "vendor/Opencv/include"
IncludeDir["Onnxruntime"] = "vendor/Onnxruntime/include"

LibsDir = {}
LibsDir["Opencv"]      = "vendor/Opencv/Lib"
LibsDir["Onnxruntime"] = "vendor/Onnxruntime/Lib"

include "vendor/GLFW"
include "vendor/Glad"

project "OBSPlaginPortraitSegmentation"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"src/**.h",
		"src/**.cpp"
	}

	defines
	{
		"_CRT_SECURE_NO_WARNINGS",
		"GLFW_INCLUDE_NONE"
	}

	includedirs
	{
		"src",
		"vendor/spdlog/include",
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
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
