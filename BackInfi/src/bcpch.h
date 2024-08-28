#pragma once
#ifndef NOMINMAX
	// See github.com/skypjack/entt/wiki/Frequently-Asked-Questions#warning-c4003-the-min-the-max-and-the-macro
	#define NOMINMAX
#endif

#include <memory>
#include <numeric>
#include <utility>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>

#include <new>
#include <mutex>

#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "BackInfi/Core/Base.h"

#ifdef BC_PLATFORM_WINDOWS
	#include <Windows.h>
#endif