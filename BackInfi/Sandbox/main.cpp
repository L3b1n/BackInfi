#include "bcpch.h"
#include "BackInfi.h"
#include "BackInfiLayer.h"

class BackInfiApp : public BackInfi::Application
{
public:
	BackInfiApp()
	{
		BackInfi::InferenceSettings settings;
		settings.UseFloatMask             = true;
		settings.EnableThreshold          = true;
		settings.EnableImageSimilarity    = true;
		settings.Threshold                = 0.9f;
		settings.ImageSimilarityThreshold = 35.0f;
		settings.TemporalSmoothFactor     = 0.85f;
		settings.Feather                  = 0.025f;
		settings.Model                    = MODEL_MEDIAPIPE;
		settings.NumThreads               = 1;

		PushLayer(new BackInfiLayer(settings));
	}
};

int main()
{
	BackInfi::Logger::Init();

	BC_PROFILE_BEGIN_SESSION("Startup", "BackInfi-Startup.json");
	auto app = new BackInfiApp();
	BC_PROFILE_END_SESSION();

	BC_PROFILE_BEGIN_SESSION("Runtime", "BackInfi-Runtime.json");
	app->Run();
	BC_PROFILE_END_SESSION();

	BC_PROFILE_BEGIN_SESSION("Shutdown", "BackInfi-Shutdown.json");
	delete app;
	BC_PROFILE_END_SESSION();

	return 0;
}