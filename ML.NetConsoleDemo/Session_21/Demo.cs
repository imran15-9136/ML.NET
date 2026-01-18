using Microsoft.ML.Data;
using Microsoft.ML;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_21
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new();


			var filePath = Path.Combine(AppContext.BaseDirectory, "Session_16", "salary_data_high_accuracy.csv");

			IDataView dataView = context.Data.LoadFromTextFile<InputModel>(path: filePath, hasHeader: true, separatorChar: ',');
			var preview = dataView.Preview();

			var shuffleData = context.Data.ShuffleRows(dataView);
			preview = shuffleData.Preview();    
			
			var skipData = context.Data.SkipRows(dataView, 8);
			preview = skipData.Preview();

			var takeData = context.Data.TakeRows(dataView, 8);
			preview = takeData.Preview();

			var filterByValue = context.Data.FilterRowsByColumn(dataView, nameof(InputModel.YearsOfExperience), lowerBound: 3, upperBound: 6);
			preview = filterByValue.Preview();

			var filterByMissingValue = context.Data.FilterRowsByMissingValues(dataView, nameof(InputModel.Salary));
			preview = filterByMissingValue.Preview();
		}
	}
}
