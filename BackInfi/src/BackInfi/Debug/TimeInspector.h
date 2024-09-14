#pragma once

#include "BackInfi/Core/Logger.h"

#include <mutex>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace BackInfi
{

	struct ProfileStats
	{
		std::string Name;
		std::chrono::duration<double, std::micro> Start;
		std::chrono::microseconds ElapsedTime;
		std::thread::id ThreadID;
	};

	struct InspectationSession
	{
		std::string Name;
	};

	class TimeInspector
	{
	public:
		TimeInspector(const TimeInspector&) = delete;
		TimeInspector(TimeInspector&&) = delete;

		void BeginSession(const std::string& name, const std::string& filepath = "results.json")
		{
			std::lock_guard lock(m_Mutex);
			if (m_CurrentSession)
			{
				if (Logger::GetCoreLogger()) // Edge case: BeginSession() might be before Logger::Init()
				{
					BC_CORE_ERROR("TimeInspector::BeginSession('{0}') when session '{1}' already open.", name, m_CurrentSession->Name);
				}
				InternalEndSession();
			}
			m_OutputStream.open(filepath);

			if (m_OutputStream.is_open())
			{
				m_CurrentSession = new InspectationSession({ name });
				WriteHeader();
			}
			else
			{
				if (Logger::GetCoreLogger()) // Edge case: BeginSession() might be before Logger::Init()
				{
					BC_CORE_ERROR("TimeInspector could not open results file '{0}'.", filepath);
				}
			}
		}

		void EndSession()
		{
			std::lock_guard lock(m_Mutex);
			InternalEndSession();
		}

		void WriteProfile(const ProfileStats& result)
		{
			std::stringstream json;

			json << std::setprecision(3) << std::fixed;
			json << ",{";
			json << "\"cat\":\"function\",";
			json << "\"dur\":" << (result.ElapsedTime.count()) << ',';
			json << "\"name\":\"" << result.Name << "\",";
			json << "\"ph\":\"X\",";
			json << "\"pid\":0,";
			json << "\"tid\":" << result.ThreadID << ",";
			json << "\"ts\":" << result.Start.count();
			json << "}";

			std::lock_guard lock(m_Mutex);
			if (m_CurrentSession)
			{
				m_OutputStream << json.str();
				m_OutputStream.flush();
			}
		}

		static TimeInspector& Get()
		{
			static TimeInspector instance;
			return instance;
		}

	private:
		TimeInspector()
			: m_CurrentSession(nullptr)
		{
		}

		~TimeInspector()
		{
			EndSession();
		}

		void WriteHeader()
		{
			m_OutputStream << "{\"otherData\": {},\"traceEvents\":[{}";
			m_OutputStream.flush();
		}

		void WriteFooter()
		{
			m_OutputStream << "]}";
			m_OutputStream.flush();
		}

		void InternalEndSession()
		{
			if (m_CurrentSession)
			{
				WriteFooter();
				m_OutputStream.close();
				delete m_CurrentSession;
				m_CurrentSession = nullptr;
			}
		}

	private:
		std::mutex           m_Mutex;
		std::ofstream        m_OutputStream;
		InspectationSession* m_CurrentSession;
	};

	class InspectationTimer
	{
	public:
		InspectationTimer(const char* name)
			: m_Name(name), m_Stopped(false)
		{
			m_StartTimepoint = std::chrono::steady_clock::now();
		}

		~InspectationTimer()
		{
			if (!m_Stopped)
				Stop();
		}

	private:
		void Stop()
		{
			auto endTimepoint = std::chrono::steady_clock::now();
			auto highResStart = std::chrono::duration<double, std::micro>{ m_StartTimepoint.time_since_epoch() };
			auto elapsedTime  = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch() -
				std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch();

			TimeInspector::Get().WriteProfile({ m_Name, highResStart, elapsedTime, std::this_thread::get_id() });

			m_Stopped = true;
		}

	private:
		bool m_Stopped;
		const char* m_Name;
		std::chrono::time_point<std::chrono::steady_clock> m_StartTimepoint;
	};

	namespace InspectorUtils
	{

		template <size_t N>
		struct ChangeResult
		{
			char Data[N];
		};

		template <size_t N, size_t K>
		constexpr auto CleanupOutputString(const char(&expr)[N], const char(&remove)[K])
		{
			ChangeResult<N> result = {};

			size_t srcIndex = 0;
			size_t dstIndex = 0;
			while (srcIndex < N)
			{
				size_t matchIndex = 0;
				while (matchIndex < K - 1 && srcIndex + matchIndex < N - 1 && expr[srcIndex + matchIndex] == remove[matchIndex])
					matchIndex++;
				if (matchIndex == K - 1)
					srcIndex += matchIndex;
				result.Data[dstIndex++] = expr[srcIndex] == '"' ? '\'' : expr[srcIndex];
				srcIndex++;
			}
			return result;
		}

	}

}

#if BC_PROFILE
	// Resolve which function signature macro will be used. Note that this only
	// is resolved when the (pre)compiler starts, so the syntax highlighting
	// could mark the wrong one in your editor!
	#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
		#define BC_FUNC_SIG __PRETTY_FUNCTION__
	#elif defined(__DMC__) && (__DMC__ >= 0x810)
		#define BC_FUNC_SIG __PRETTY_FUNCTION__
	#elif (defined(__FUNCSIG__) || (_MSC_VER))
		#define BC_FUNC_SIG __FUNCSIG__
	#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
		#define BC_FUNC_SIG __FUNCTION__
	#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
		#define BC_FUNC_SIG __FUNC__
	#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
		#define BC_FUNC_SIG __func__
	#elif defined(__cplusplus) && (__cplusplus >= 201103)
		#define BC_FUNC_SIG __func__
	#else
		#define BC_FUNC_SIG "BC_FUNC_SIG unknown!"
	#endif

	#define BC_PROFILE_BEGIN_SESSION(name, filepath) ::BackInfi::TimeInspector::Get().BeginSession(name, filepath)
	#define BC_PROFILE_END_SESSION()                 ::BackInfi::TimeInspector::Get().EndSession()
	#define BC_PROFILE_SCOPE_LINE2(name, line)       constexpr auto fixedName##line = ::BackInfi::InspectorUtils::CleanupOutputString(name, "__cdecl ");\
												     ::BackInfi::InspectationTimer timer##line(fixedName##line.Data)

	#define BC_PROFILE_SCOPE_LINE(name, line) BC_PROFILE_SCOPE_LINE2(name, line)
	#define BC_PROFILE_SCOPE(name)            BC_PROFILE_SCOPE_LINE(name, __LINE__)
	#define BC_PROFILE_FUNC()                 BC_PROFILE_SCOPE(BC_FUNC_SIG)
#else
	#define BC_PROFILE_BEGIN_SESSION(name, filepath)
	#define BC_PROFILE_END_SESSION()
	#define BC_PROFILE_SCOPE(name)
	#define BC_PROFILE_FUNC()
#endif