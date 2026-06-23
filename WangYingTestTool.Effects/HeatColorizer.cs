using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Effects;

namespace WangYingTestTool.Effects;

public class HeatColorizer : ShaderEffect
{
	public static readonly DependencyProperty InputProperty = ShaderEffect.RegisterPixelShaderSamplerProperty("Input", typeof(HeatColorizer), 0, SamplingMode.Auto);

	public static readonly DependencyProperty PaletteProperty = ShaderEffect.RegisterPixelShaderSamplerProperty("Palette", typeof(HeatColorizer), 1, SamplingMode.Auto);

	public virtual Brush Input
	{
		get
		{
			return (Brush)GetValue(InputProperty);
		}
		set
		{
			SetValue(InputProperty, value);
		}
	}

	public virtual Brush Palette
	{
		get
		{
			return (Brush)GetValue(PaletteProperty);
		}
		set
		{
			SetValue(PaletteProperty, value);
		}
	}

	public HeatColorizer()
	{
		base.PixelShader = new PixelShader
		{
			UriSource = new Uri("/WangYingTestTool;component/Effects/HeatColorizer.ps", UriKind.Relative)
		};
		UpdateShaderValue(InputProperty);
		UpdateShaderValue(PaletteProperty);
	}
}
