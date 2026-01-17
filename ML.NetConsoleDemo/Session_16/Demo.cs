using Microsoft.ML;
using Microsoft.ML.Data;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_16
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new();

			var columns = new TextLoader.Column[]
			{
				new TextLoader.Column("YearsOfExperience", DataKind.Single, 0),
				new TextLoader.Column("Salary", DataKind.Single, 1),
			};

			var filePath = Path.Combine(AppContext.BaseDirectory,"Session_16", "salary_data_high_accuracy.tsv");

			//IDataView dataView = context.Data.LoadFromTextFile<InputModel>(path: filePath, hasHeader:true, separatorChar: ',');

			IDataView dataView = context.Data.LoadFromTextFile<InputModel>(path: filePath, hasHeader: false);


			var preview = dataView.Preview();
		}
	}
}
