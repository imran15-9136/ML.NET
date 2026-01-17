using Microsoft.ML;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_14
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new();

			var samples = new List<InputModel>
			{
			new InputModel { YearsOfExperience = 1.0f, Salary = 31000 },
			new InputModel { YearsOfExperience = 1.5f, Salary = 34000 },
			new InputModel { YearsOfExperience = 2.0f, Salary = 37000 },
			new InputModel { YearsOfExperience = 2.5f, Salary = 40000 },
			new InputModel { YearsOfExperience = 3.0f, Salary = 43000 },
			new InputModel { YearsOfExperience = 3.5f, Salary = 46000 },
			new InputModel { YearsOfExperience = 4.0f, Salary = 49000 },
			new InputModel { YearsOfExperience = 4.5f, Salary = 52000 },
			new InputModel { YearsOfExperience = 5.0f, Salary = 55000 },
			new InputModel { YearsOfExperience = 5.5f, Salary = 58000 },

			new InputModel { YearsOfExperience = 6.0f, Salary = 61000 },
			new InputModel { YearsOfExperience = 6.5f, Salary = 64000 },
			new InputModel { YearsOfExperience = 7.0f, Salary = 67000 },
			new InputModel { YearsOfExperience = 7.5f, Salary = 70000 },
			new InputModel { YearsOfExperience = 8.0f, Salary = 73000 },
			new InputModel { YearsOfExperience = 8.5f, Salary = 76000 },
			new InputModel { YearsOfExperience = 9.0f, Salary = 79000 },
			new InputModel { YearsOfExperience = 9.5f, Salary = 82000 },
			new InputModel { YearsOfExperience = 10.0f, Salary = 85000 },
			new InputModel { YearsOfExperience = 10.5f, Salary = 88000 },

			new InputModel { YearsOfExperience = 11.0f, Salary = 91000 },
			new InputModel { YearsOfExperience = 11.5f, Salary = 94000 },
			new InputModel { YearsOfExperience = 12.0f, Salary = 97000 },
			new InputModel { YearsOfExperience = 12.5f, Salary = 100000 },
			new InputModel { YearsOfExperience = 13.0f, Salary = 103000 },
			new InputModel { YearsOfExperience = 13.5f, Salary = 106000 },
			new InputModel { YearsOfExperience = 14.0f, Salary = 109000 },
			new InputModel { YearsOfExperience = 14.5f, Salary = 112000 },
			new InputModel { YearsOfExperience = 15.0f, Salary = 115000 },
			new InputModel { YearsOfExperience = 15.5f, Salary = 118000 },

			new InputModel { YearsOfExperience = 16.0f, Salary = 121000 },
			new InputModel { YearsOfExperience = 16.5f, Salary = 124000 },
			new InputModel { YearsOfExperience = 17.0f, Salary = 127000 },
			new InputModel { YearsOfExperience = 17.5f, Salary = 130000 },
			new InputModel { YearsOfExperience = 18.0f, Salary = 133000 },
			new InputModel { YearsOfExperience = 18.5f, Salary = 136000 },
			new InputModel { YearsOfExperience = 19.0f, Salary = 139000 },
			new InputModel { YearsOfExperience = 19.5f, Salary = 142000 },
			new InputModel { YearsOfExperience = 20.0f, Salary = 145000 }
			};


			IDataView traningData = context.Data.LoadFromEnumerable(samples);

			var estimator = context.Transforms.Concatenate("Features", nameof(InputModel.YearsOfExperience));

			//var pipeline = estimator.Append(context.Regression.Trainers.Sdca(nameof(InputModel.Salary), maximumNumberOfIterations: 10, l1Regularization:2,l2Regularization:2));

			var pipeline = estimator.Append(context.Regression.Trainers.OnlineGradientDescent(nameof(InputModel.Salary), numberOfIterations:100));


			var model = pipeline.Fit(traningData);

			var testData = new List<InputModel>
			{
				new InputModel { YearsOfExperience = 1, Salary = 28000 },
				new InputModel { YearsOfExperience = 2.5F, Salary = 37000 },
				new InputModel { YearsOfExperience = 3, Salary = 40000 },
				new InputModel { YearsOfExperience = 3.2F, Salary = 42000 },
				new InputModel { YearsOfExperience = 4.7F, Salary = 47000 },
				new InputModel { YearsOfExperience = 7, Salary = 60000 },
			};

			var testDataView = context.Data.LoadFromEnumerable(testData);

			var metrics = context.Regression.Evaluate(model.Transform(testDataView), nameof(InputModel.Salary));

			Console.WriteLine($"R²: {metrics.RSquared:0.##}");
			
		}
	}
}
