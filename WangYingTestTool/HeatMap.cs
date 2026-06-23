using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;

namespace WangYingTestTool;

public class HeatMap : FrameworkElement
{
	private const int GradientNumber = 256;

	private VisualCollection m_heat_map_visuals;

	private List<HeatPoint> m_heat_points = new List<HeatPoint>();

	private readonly RadialGradientBrush[] m_brushes;

	protected override int VisualChildrenCount => m_heat_map_visuals.Count;

	public HeatMap()
	{
		m_heat_map_visuals = new VisualCollection(this);
		m_brushes = new RadialGradientBrush[256];
		for (int i = 0; i < m_brushes.Length; i++)
		{
			m_brushes[i] = new RadialGradientBrush(Color.FromArgb((byte)i, byte.MaxValue, byte.MaxValue, byte.MaxValue), Color.FromArgb(0, byte.MaxValue, byte.MaxValue, byte.MaxValue));
		}
	}

	protected override Visual GetVisualChild(int index)
	{
		if (index < 0 || index >= m_heat_map_visuals.Count)
		{
			throw new ArgumentOutOfRangeException("index");
		}
		return m_heat_map_visuals[index];
	}

	public void AddHeatPoint(HeatPoint in_heat_point)
	{
		m_heat_points.Add(in_heat_point);
	}

	public void Clear()
	{
		m_heat_points.Clear();
	}

	public void Render()
	{
		m_heat_map_visuals.Clear();
		DrawingVisual drawing_visual = new DrawingVisual();
		using DrawingContext dc = drawing_visual.RenderOpen();
		foreach (HeatPoint point in m_heat_points)
		{
			double width_height = ((point.Intensity >= 30) ? ((point.Intensity >= 60) ? ((point.Intensity >= 90) ? ((point.Intensity >= 120) ? ((double)(point.Intensity / 2)) : ((double)(int)point.Intensity * 0.75)) : ((double)(int)point.Intensity)) : ((double)(int)point.Intensity * 1.5)) : ((double)(point.Intensity * 2)));
			dc.DrawRectangle(m_brushes[point.Intensity], null, new Rect((double)point.X - width_height / 2.0, (double)point.Y - width_height / 2.0, width_height, width_height));
		}
		m_heat_map_visuals.Add(drawing_visual);
	}
}
