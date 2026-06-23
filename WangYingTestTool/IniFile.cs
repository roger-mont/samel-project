using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace WangYingTestTool;

internal class IniFile
{
	private string _pathIni;

	public string PathIni
	{
		get
		{
			return _pathIni;
		}
		set
		{
			_pathIni = value;
		}
	}

	[DllImport("kernel32.dll", SetLastError = true)]
	public static extern long WritePrivateProfileString(string section, string key, string val, string filePath);

	[DllImport("kernel32.dll", SetLastError = true)]
	public static extern int GetPrivateProfileString(string section, string key, string def, StringBuilder retVal, int size, string filePath);

	public IniFile(string path)
	{
		_pathIni = path;
	}

	public bool IniFileExist()
	{
		return File.Exists(_pathIni);
	}

	public bool CreateIniFile()
	{
		FileStream fs = File.Create(_pathIni);
		fs.Close();
		fs.Dispose();
		return IniFileExist();
	}

	public void IniWriteValue(string Section, string Key, string Value)
	{
		WritePrivateProfileString(Section, Key, Value, _pathIni);
	}

	public string IniReadValue(string Section, string Key)
	{
		StringBuilder temp = new StringBuilder(500);
		int i = GetPrivateProfileString(Section, Key, "", temp, 500, _pathIni);
		return temp.ToString();
	}
}
