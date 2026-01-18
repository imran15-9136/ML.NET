using Microsoft.ML;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_17
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new();

			var filePath = Path.Combine(AppContext.BaseDirectory, "Session_17", "traning-dataset","*");

			IDataView dataView = context.Data.LoadFromTextFile<InputModel>(path: filePath, hasHeader: false, separatorChar: ',');

			var preview = dataView.Preview();
		}
	}
}
