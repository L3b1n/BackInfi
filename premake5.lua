workspace "OBSPlaginPortraitSegmentation"
	startproject "OBSPlaginPortraitSegmentation"
	architecture "x64"

	configurations
	{
		"Debug",
		"Release",
		"Dist"
	}

	flags
	{
		"MultiProcessorCompile"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "OBSPlaginPortraitSegmentation"
