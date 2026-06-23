using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Threading;
using UsbHid;

namespace WangYingTestTool;

public partial class MainWindow : Window, IComponentConnector
{
	private delegate void MODIFYVISIBLE(byte[] data);

	private delegate void MODIFYVISIBLEV(DataTable dt);

	private DispatcherTimer ShowTimer;

	private string time_str = "";

	private string path_he = Environment.CurrentDirectory + "\\File_ini.ini";

	private IniFile heIniFile = null;

	private UsbHidDevice uhid = null;

	private Thread thMsgRecivce = null;

	private DataTable dg;

	private bool Insertbool = false;

	private long rec_num = 0L;

	private int[,] dataDouble = new int[64, 16];

	private int[,] MapV = new int[64, 64];

	private bool dataEx = false;

	private string exdata_str = "";

	private long numdata = 0L;

	private bool wir = false;

	private int[,] xwhng1 = new int[256, 2]
	{
		{ 15, 15 },
		{ 15, 14 },
		{ 15, 13 },
		{ 15, 12 },
		{ 15, 11 },
		{ 15, 10 },
		{ 15, 9 },
		{ 15, 8 },
		{ 15, 7 },
		{ 15, 6 },
		{ 15, 5 },
		{ 15, 4 },
		{ 15, 3 },
		{ 15, 2 },
		{ 15, 1 },
		{ 15, 0 },
		{ 14, 15 },
		{ 14, 14 },
		{ 14, 13 },
		{ 14, 12 },
		{ 14, 11 },
		{ 14, 10 },
		{ 14, 9 },
		{ 14, 8 },
		{ 14, 7 },
		{ 14, 6 },
		{ 14, 5 },
		{ 14, 4 },
		{ 14, 3 },
		{ 14, 2 },
		{ 14, 1 },
		{ 14, 0 },
		{ 13, 15 },
		{ 13, 14 },
		{ 13, 13 },
		{ 13, 12 },
		{ 13, 11 },
		{ 13, 10 },
		{ 13, 9 },
		{ 13, 8 },
		{ 13, 7 },
		{ 13, 6 },
		{ 13, 5 },
		{ 13, 4 },
		{ 13, 3 },
		{ 13, 2 },
		{ 13, 1 },
		{ 13, 0 },
		{ 12, 15 },
		{ 12, 14 },
		{ 12, 13 },
		{ 12, 12 },
		{ 12, 11 },
		{ 12, 10 },
		{ 12, 9 },
		{ 12, 8 },
		{ 12, 7 },
		{ 12, 6 },
		{ 12, 5 },
		{ 12, 4 },
		{ 12, 3 },
		{ 12, 2 },
		{ 12, 1 },
		{ 12, 0 },
		{ 11, 15 },
		{ 11, 14 },
		{ 11, 13 },
		{ 11, 12 },
		{ 11, 11 },
		{ 11, 10 },
		{ 11, 9 },
		{ 11, 8 },
		{ 11, 7 },
		{ 11, 6 },
		{ 11, 5 },
		{ 11, 4 },
		{ 11, 3 },
		{ 11, 2 },
		{ 11, 1 },
		{ 11, 0 },
		{ 10, 15 },
		{ 10, 14 },
		{ 10, 13 },
		{ 10, 12 },
		{ 10, 11 },
		{ 10, 10 },
		{ 10, 9 },
		{ 10, 8 },
		{ 10, 7 },
		{ 10, 6 },
		{ 10, 5 },
		{ 10, 4 },
		{ 10, 3 },
		{ 10, 2 },
		{ 10, 1 },
		{ 10, 0 },
		{ 9, 15 },
		{ 9, 14 },
		{ 9, 13 },
		{ 9, 12 },
		{ 9, 11 },
		{ 9, 10 },
		{ 9, 9 },
		{ 9, 8 },
		{ 9, 7 },
		{ 9, 6 },
		{ 9, 5 },
		{ 9, 4 },
		{ 9, 3 },
		{ 9, 2 },
		{ 9, 1 },
		{ 9, 0 },
		{ 8, 15 },
		{ 8, 14 },
		{ 8, 13 },
		{ 8, 12 },
		{ 8, 11 },
		{ 8, 10 },
		{ 8, 9 },
		{ 8, 8 },
		{ 8, 7 },
		{ 8, 6 },
		{ 8, 5 },
		{ 8, 4 },
		{ 8, 3 },
		{ 8, 2 },
		{ 8, 1 },
		{ 8, 0 },
		{ 7, 15 },
		{ 7, 14 },
		{ 7, 13 },
		{ 7, 12 },
		{ 7, 11 },
		{ 7, 10 },
		{ 7, 9 },
		{ 7, 8 },
		{ 7, 7 },
		{ 7, 6 },
		{ 7, 5 },
		{ 7, 4 },
		{ 7, 3 },
		{ 7, 2 },
		{ 7, 1 },
		{ 7, 0 },
		{ 6, 15 },
		{ 6, 14 },
		{ 6, 13 },
		{ 6, 12 },
		{ 6, 11 },
		{ 6, 10 },
		{ 6, 9 },
		{ 6, 8 },
		{ 6, 7 },
		{ 6, 6 },
		{ 6, 5 },
		{ 6, 4 },
		{ 6, 3 },
		{ 6, 2 },
		{ 6, 1 },
		{ 6, 0 },
		{ 5, 15 },
		{ 5, 14 },
		{ 5, 13 },
		{ 5, 12 },
		{ 5, 11 },
		{ 5, 10 },
		{ 5, 9 },
		{ 5, 8 },
		{ 5, 7 },
		{ 5, 6 },
		{ 5, 5 },
		{ 5, 4 },
		{ 5, 3 },
		{ 5, 2 },
		{ 5, 1 },
		{ 5, 0 },
		{ 4, 15 },
		{ 4, 14 },
		{ 4, 13 },
		{ 4, 12 },
		{ 4, 11 },
		{ 4, 10 },
		{ 4, 9 },
		{ 4, 8 },
		{ 4, 7 },
		{ 4, 6 },
		{ 4, 5 },
		{ 4, 4 },
		{ 4, 3 },
		{ 4, 2 },
		{ 4, 1 },
		{ 4, 0 },
		{ 3, 15 },
		{ 3, 14 },
		{ 3, 13 },
		{ 3, 12 },
		{ 3, 11 },
		{ 3, 10 },
		{ 3, 9 },
		{ 3, 8 },
		{ 3, 7 },
		{ 3, 6 },
		{ 3, 5 },
		{ 3, 4 },
		{ 3, 3 },
		{ 3, 2 },
		{ 3, 1 },
		{ 3, 0 },
		{ 2, 15 },
		{ 2, 14 },
		{ 2, 13 },
		{ 2, 12 },
		{ 2, 11 },
		{ 2, 10 },
		{ 2, 9 },
		{ 2, 8 },
		{ 2, 7 },
		{ 2, 6 },
		{ 2, 5 },
		{ 2, 4 },
		{ 2, 3 },
		{ 2, 2 },
		{ 2, 1 },
		{ 2, 0 },
		{ 1, 15 },
		{ 1, 14 },
		{ 1, 13 },
		{ 1, 12 },
		{ 1, 11 },
		{ 1, 10 },
		{ 1, 9 },
		{ 1, 8 },
		{ 1, 7 },
		{ 1, 6 },
		{ 1, 5 },
		{ 1, 4 },
		{ 1, 3 },
		{ 1, 2 },
		{ 1, 1 },
		{ 1, 0 },
		{ 0, 15 },
		{ 0, 14 },
		{ 0, 13 },
		{ 0, 12 },
		{ 0, 11 },
		{ 0, 10 },
		{ 0, 9 },
		{ 0, 8 },
		{ 0, 7 },
		{ 0, 6 },
		{ 0, 5 },
		{ 0, 4 },
		{ 0, 3 },
		{ 0, 2 },
		{ 0, 1 },
		{ 0, 0 }
	};

	private int[,] ywhng = new int[16, 16];

	private Queue<int> quedata_x = new Queue<int>();

	private Queue<int> quedata_y = new Queue<int>();

	private Queue<int> quedata_data = new Queue<int>();

	private double widths = 20.0;

	private double heights = 18.0;

	private DispatcherTimer disTimer;

	private DispatcherTimer disTimerA;

	private DispatcherTimer disTimerAD;

	private int countSecond = 0;

	private StreamWriter zongsw;

	private FileStream zongfs;

	private Queue<byte[]> queRecDat = new Queue<byte[]>();

	private DispatcherTimer showTimer = new DispatcherTimer();

	private bool IsBufAvailabe => queRecDat.Count > 0;

	public MainWindow()
	{
		InitializeComponent();
		time_str = DateTime.Now.ToString("yyyy-MM-dd-mm-ss");
		ShowTimer = new DispatcherTimer();
		ShowTimer.Tick += ShowCurTimer;
		ShowTimer.Interval = new TimeSpan(0, 0, 0, 1, 0);
		cbColors.SelectedIndex = 0;
	}

	public void ShowCurTimer(object sender, EventArgs e)
	{
		time_str = DateTime.Now.ToString("yyyy-MM-dd HH:mm");
	}

	private void Window_Loaded(object sender, RoutedEventArgs e)
	{
		DtaLoge();
		if (File.Exists(path_he))
		{
			if (uhid == null)
			{
				uhid = new UsbHidDevice(6860, 6733);
				uhid.DataReceived += DeviceDataReceived;
				uhid.Connect();
				thMsgRecivce = new Thread(HidMsgDeal);
				thMsgRecivce.Start();
			}
			CountDown2();
		}
	}

	private void DtaLoge()
	{
		try
		{
			dg = new DataTable();
			for (int i = 0; i < 64; i++)
			{
				dg.Columns.Add(i.ToString(), typeof(int));
			}
			int number = Convert.ToInt32(31);
			if (dataGrid.Items.Count > 0)
			{
				dg = ((DataView)dataGrid.ItemsSource).Table;
				int j = dg.Rows.Count;
				for (int k = j; k < j + number; k++)
				{
					dg.Rows.Add();
					dg.Rows[k][0] = k;
				}
				dataGrid.ItemsSource = dg.DefaultView;
			}
			else
			{
				for (int l = 0; l <= number; l++)
				{
					dg.Rows.Add();
					dg.Rows[l][0] = l;
				}
				dataGrid.ItemsSource = dg.DefaultView;
			}
		}
		catch (Exception)
		{
			throw;
		}
	}

	private void Insert_Click(object sender, RoutedEventArgs e)
	{
		Insertbool = true;
		cHeatMap.Visibility = Visibility.Hidden;
		dataGrid.Visibility = Visibility.Visible;
	}

	private void export_Click(object sender, RoutedEventArgs e)
	{
		countSecond = Convert.ToInt32(textBox.Text);
		dataEx = true;
		CountDown23();
		CountDown();
		SaveCSV(dg);
		rec_num = Convert.ToInt32(textBox.Text);
		label1.Content = rec_num;
	}

	private void button1_Click(object sender, RoutedEventArgs e)
	{
		countSecond = 3;
		SaveCSV3S(dg);
		dataEx = true;
		CountDown23();
		CountDown();
		rec_num = Convert.ToInt32(textBox.Text);
		label1.Content = rec_num;
	}

	public void DeviceDataReceived(byte[] data)
	{
		AddRecDat(data);
	}

	private void HidMsgDeal()
	{
		while (true)
		{
			if (IsBufAvailabe)
			{
				byte[] rec_data = GetRecDat();
				HidReciveProg(rec_data);
			}
			else
			{
				Thread.Sleep(1);
			}
		}
	}

	private void HidReciveProg(byte[] data)
	{
		base.Dispatcher.Invoke(new MODIFYVISIBLE(ButtonShowDeal), data);
	}

	public void ButtonShowDeal(byte[] data)
	{
		Random rRand = new Random();
		int x = 0;
		int y = 0;
		int v = 0;
		for (int i = 1; i < data.Length - 1; i += 3)
		{
			int x_2 = data[i];
			int y_2 = data[i + 1];
			byte v_2 = data[i + 2];
			if (data[0] == 0 || x_2 == 0 || y_2 == 0)
			{
				break;
			}
			if (data[0] < 5)
			{
				x = 16 * ((9 - data[0]) / 5) + (16 - x_2);
				y = 16 * (data[0] - 1) + (16 - y_2);
				MapV[x, y] = v_2;
			}
			else if (data[0] < 9)
			{
				x = 16 * ((9 - data[0]) / 5) + (x_2 - 1);
				y = 16 * (8 - data[0]) + (y_2 - 1);
				MapV[x, y] = v_2;
			}
		}
	}

	public void CountDown()
	{
		disTimer = new DispatcherTimer();
		disTimer.Tick += disTimer_Tick;
		disTimer.Interval = new TimeSpan(9200000L);
		disTimer.Start();
	}

	public void CountDown2()
	{
		disTimerA = new DispatcherTimer();
		disTimerA.Tick += disTimer2_Tick;
		disTimerA.Interval = new TimeSpan(250000L);
		disTimerA.Start();
	}

	public void CountDown23()
	{
		disTimerAD = new DispatcherTimer();
		disTimerAD.Tick += disTimer3_Tick;
		disTimerAD.Interval = new TimeSpan(100000L);
		disTimerAD.Start();
	}

	private void disTimer3_Tick(object sender, EventArgs e)
	{
		if (!dataEx)
		{
			return;
		}
		for (int i = 0; i < dg.Rows.Count; i++)
		{
			exdata_str = "";
			for (int j = 0; j < dg.Columns.Count; j++)
			{
				string str = MapV[i, j].ToString();
				exdata_str += str;
				if (j < dg.Columns.Count - 1)
				{
					exdata_str += ",";
				}
			}
			zongsw.WriteLine(exdata_str);
		}
		zongsw.WriteLine("");
		wir = true;
	}

	private void disTimer_Tick(object sender, EventArgs e)
	{
		if (countSecond == 0)
		{
			if (wir)
			{
				export.IsEnabled = true;
				dataEx = false;
				zongsw.Close();
				zongfs.Close();
				wir = false;
				MessageBox.Show("Export complete!");
				disTimer.Stop();
				disTimerAD.Stop();
			}
		}
		else
		{
			countSecond--;
		}
	}

	private void disTimer2_Tick(object sender, EventArgs e)
	{
		cHeatMap.Clear();
		for (int d_x = 0; d_x < 32; d_x++)
		{
			exdata_str = "";
			for (int d_y = 0; d_y < 64; d_y++)
			{
				dg.Rows[d_x][d_y] = MapV[d_x, d_y];
				DataGridRow row = (DataGridRow)dataGrid.ItemContainerGenerator.ContainerFromIndex(d_x);
				DataGridCellsPresenter presenter = GetVisualChild<DataGridCellsPresenter>(row);
				DataGridCell cell = (DataGridCell)presenter.ItemContainerGenerator.ContainerFromIndex(d_y);
				if (Insertbool)
				{
					if (MapV[d_x, d_y] < 26)
					{
						cell.Background = new SolidColorBrush(Colors.White);
					}
					else if (MapV[d_x, d_y] >= 26 && MapV[d_x, d_y] < 77)
					{
						cell.Background = new SolidColorBrush(Color.FromArgb(byte.MaxValue, 210, 245, 242));
					}
					else if (MapV[d_x, d_y] >= 77 && MapV[d_x, d_y] < 115)
					{
						cell.Background = new SolidColorBrush(Color.FromArgb(byte.MaxValue, 161, 245, 161));
					}
					else if (MapV[d_x, d_y] >= 115 && MapV[d_x, d_y] < 141)
					{
						cell.Background = new SolidColorBrush(Color.FromArgb(byte.MaxValue, byte.MaxValue, byte.MaxValue, 0));
					}
					else if (MapV[d_x, d_y] >= 141 && MapV[d_x, d_y] < 154)
					{
						cell.Background = new SolidColorBrush(Color.FromArgb(byte.MaxValue, byte.MaxValue, 138, 0));
					}
					else if (MapV[d_x, d_y] >= 154 && MapV[d_x, d_y] < 256)
					{
						cell.Background = new SolidColorBrush(Color.FromArgb(byte.MaxValue, byte.MaxValue, 0, 0));
					}
				}
				if (MapV[d_x, d_y] > 0)
				{
					cHeatMap.AddHeatPoint(new HeatPoint((int)((double)(d_y + 1) * widths), (int)((double)(d_x + 1) * heights), (byte)MapV[d_x, d_y]));
				}
			}
		}
		if (dataEx)
		{
			for (int i = 0; i < dg.Rows.Count; i++)
			{
				exdata_str = "";
				for (int j = 0; j < dg.Columns.Count; j++)
				{
					string str = dg.Rows[i][j].ToString();
					exdata_str += str;
					if (j < dg.Columns.Count - 1)
					{
						exdata_str += ",";
					}
				}
			}
		}
		dataGrid.ItemsSource = dg.DefaultView;
		cHeatMap.Render();
	}

	public void SaveCSV(DataTable dt)
	{
		export.IsEnabled = false;
		string str_1 = AppDomain.CurrentDomain.BaseDirectory + "data";
		if (txt_filename.Text != "")
		{
			str_1 = str_1 + "\\" + txt_filename.Text + ".csv";
			string strFormat = str_1;
			FileInfo fi = new FileInfo(strFormat);
			if (!fi.Directory.Exists)
			{
				fi.Directory.Create();
			}
			StreamWriter sw = new StreamWriter(zongfs = new FileStream(strFormat, FileMode.Create, FileAccess.Write), Encoding.UTF8);
			zongsw = sw;
			string data = "";
		}
		else
		{
			dataEx = false;
			export.IsEnabled = true;
			MessageBox.Show("File name is not filled in!");
		}
	}

	public void SaveCSV3S(DataTable dt)
	{
		export.IsEnabled = false;
		string str_1 = AppDomain.CurrentDomain.BaseDirectory + "data";
		if (txt_filename.Text != "")
		{
			str_1 = str_1 + "\\ZERO_" + txt_filename.Text + ".csv";
			string strFormat = str_1;
			FileInfo fi = new FileInfo(strFormat);
			if (!fi.Directory.Exists)
			{
				fi.Directory.Create();
			}
			StreamWriter sw = new StreamWriter(zongfs = new FileStream(strFormat, FileMode.Create, FileAccess.Write), Encoding.UTF8);
			zongsw = sw;
			string data = "";
		}
		else
		{
			dataEx = false;
			export.IsEnabled = true;
			MessageBox.Show("File name is not filled in!");
		}
	}

	public static T GetVisualChild<T>(Visual parent) where T : Visual
	{
		T childContent = null;
		int numVisuals = VisualTreeHelper.GetChildrenCount(parent);
		for (int i = 0; i < numVisuals; i++)
		{
			Visual v = (Visual)VisualTreeHelper.GetChild(parent, i);
			childContent = v as T;
			if (childContent == null)
			{
				childContent = GetVisualChild<T>(v);
			}
			if (childContent != null)
			{
				break;
			}
		}
		return childContent;
	}

	private DataGrid DynamicCreationDGV(DataGrid dgv)
	{
		dgv.Columns.Clear();
		return dgv;
	}

	private void AddRecDat(byte[] data)
	{
		if (data != null)
		{
			lock (this)
			{
				queRecDat.Enqueue(data);
			}
		}
	}

	private byte[] GetRecDat()
	{
		byte[] temp_dat = null;
		lock (this)
		{
			temp_dat = queRecDat.Dequeue();
		}
		return temp_dat;
	}

	private void Grid_SizeChanged(object sender, SizeChangedEventArgs e)
	{
	}

	private void RenderContent()
	{
		cHeatMap.Clear();
		Random rRand = new Random();
		for (int i = 0; i < 256; i++)
		{
			int x = rRand.Next(0, (int)(cHeatMap.ActualWidth - 1.0));
			int y = rRand.Next(0, (int)(cHeatMap.ActualHeight - 1.0));
			byte intense = (byte)rRand.Next(0, 255);
			cHeatMap.AddHeatPoint(new HeatPoint(x, y, intense));
		}
		cHeatMap.Render();
	}

	private void SetNull(object sender, EventArgs e)
	{
	}

	private void button_Click(object sender, RoutedEventArgs e)
	{
		Insertbool = false;
		cHeatMap.Visibility = Visibility.Visible;
		dataGrid.Visibility = Visibility.Hidden;
		label.Visibility = Visibility.Hidden;
	}

	private void Window_Closed(object sender, EventArgs e)
	{
		Process.GetCurrentProcess().Kill();
	}

	private void button2_Click(object sender, RoutedEventArgs e)
	{
		string str_1 = AppDomain.CurrentDomain.BaseDirectory + "data\\test.csv";
		string localFilePath = str_1;
		string fileNameInS3 = "test.csv";
		AmazonS3Uploader amazonS3 = new AmazonS3Uploader();
		amazonS3.UploadFileToS3(localFilePath, fileNameInS3);
	}

	[DebuggerNonUserCode]
	[GeneratedCode("PresentationBuildTasks", "4.0.0.0")]
	internal Delegate _CreateDelegate(Type delegateType, string handler)
	{
		return Delegate.CreateDelegate(delegateType, this, handler);
	}
}
