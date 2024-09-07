#include "bcpch.h"

#include "BackInfi/Core/Layer.h"

namespace BackInfi
{

	// Layer
	// ------------------------------------------------------------------------
	Layer::Layer(const std::string& name)
	{
		m_DebugName = name;
	}
	
	// LayerStack
	// ------------------------------------------------------------------------
	LayerStack::~LayerStack()
	{
		for (auto layer : m_Layers)
		{
			layer->OnDetach();
			delete layer;
		}
	}

	void LayerStack::PushLayer(Layer* layer)
	{
		m_Layers.emplace(m_Layers.begin() + m_LayerIndex, layer);
		m_LayerIndex++;
	}

	void LayerStack::PushOverlay(Layer* overlay)
	{
		m_Layers.emplace_back(overlay);
	}

	void LayerStack::PopLayer(Layer* layer)
	{
		auto iterator = std::find(m_Layers.begin(), m_Layers.begin() + m_LayerIndex, layer);
		if (iterator != m_Layers.begin() + m_LayerIndex)
		{
			layer->OnDetach();
			m_Layers.erase(iterator);
			m_LayerIndex--;
		}
	}

	void LayerStack::PopOverlay(Layer* overlay)
	{
		auto iterator = std::find(m_Layers.begin() + m_LayerIndex, m_Layers.end(), overlay);
		if (iterator != m_Layers.end())
		{
			overlay->OnDetach();
			m_Layers.erase(iterator);
		}
	}

}