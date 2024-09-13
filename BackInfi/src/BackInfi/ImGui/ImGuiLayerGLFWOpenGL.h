#pragma once

#include "BackInfi/Core/Layer.h"

#include "BackInfi/Events/Event.h"

namespace BackInfi
{

	class ImGuiLayerGLFWOpenGL : public Layer
	{
	public:
		ImGuiLayerGLFWOpenGL();
		~ImGuiLayerGLFWOpenGL();

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnEvent(Event& e) override;

		void Begin();
		void End();

		uint32_t GetActiveWidetID() const;

	private:
		void SetDarkThemeColors();
	};

}