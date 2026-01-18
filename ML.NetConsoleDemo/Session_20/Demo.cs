using Microsoft.ML;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_20
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new();

			var filePath = Path.Combine(AppContext.BaseDirectory, "Session_17", "traning-dataset", "*");

			var baseDir = Path.Combine(AppContext.BaseDirectory, "Session_20", "traning-dataset");

			if(!Directory.Exists(baseDir))
				Directory.CreateDirectory(baseDir);

			var savePath = Path.Combine(baseDir, "combined-dataset.bin");

			IDataView dataView = context.Data.LoadFromTextFile<InputModel>(path: filePath, hasHeader: false, separatorChar: ',');

			var list = context.Data.CreateEnumerable<InputModel>(dataView, reuseRowObject: false).ToList();

			// Save combined dataset
			using FileStream stream = new(savePath, FileMode.Create);
			//context.Data.SaveAsText(dataView, stream, separatorChar: ' ');
			context.Data.SaveAsBinary(dataView, stream);
		}
	}
}
