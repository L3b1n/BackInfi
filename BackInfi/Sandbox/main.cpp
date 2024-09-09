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

	auto app = new BackInfiApp();
	app->Run();
	delete app;

	return 0;
}