using Microsoft.ML;
using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_10
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new ();

			var samples = new List<InputModel>
			{
				new InputModel { YearsOfExperience = 1, Salary = 28000 },
				new InputModel { YearsOfExperience = 1, Salary = 30000 },
				new InputModel { YearsOfExperience = 1.3F, Salary = 32000 },
				new InputModel { YearsOfExperience = 2, Salary = 33000 },
				new InputModel { YearsOfExperience = 2, Salary = 35000 },
				new InputModel { YearsOfExperience = 2.5F, Salary = 37000 },
				new InputModel { YearsOfExperience = 3, Salary = 38000 },
				new InputModel { YearsOfExperience = 3, Salary = 40000 },
				new InputModel { YearsOfExperience = 3.2F, Salary = 42000 },
				new InputModel { YearsOfExperience = 4, Salary = 43000 },
				new InputModel { YearsOfExperience = 4, Salary = 45000 },
				new InputModel { YearsOfExperience = 4.7F, Salary = 47000 },
				new InputModel { YearsOfExperience = 5, Salary = 48000 },
				new InputModel { YearsOfExperience = 5.2F, Salary = 50000 },
				new InputModel { YearsOfExperience = 5.5F, Salary = 52000 },

				new InputModel { YearsOfExperience = 6, Salary = 54000 },
				new InputModel { YearsOfExperience = 6.4F, Salary = 56000 },
				new InputModel { YearsOfExperience = 6.6F, Salary = 58000 },
				new InputModel { YearsOfExperience = 7, Salary = 60000 },
				new InputModel { YearsOfExperience = 7.2F, Salary = 62000 },
				new InputModel { YearsOfExperience = 7.6F, Salary = 65000 },
				new InputModel { YearsOfExperience = 8, Salary = 67000 },
				new InputModel { YearsOfExperience = 8, Salary = 70000 },
				new InputModel { YearsOfExperience = 8, Salary = 73000 },
				new InputModel { YearsOfExperience = 9, Salary = 75000 },
				new InputModel { YearsOfExperience = 9.2F, Salary = 78000 },
				new InputModel { YearsOfExperience = 9.5F, Salary = 81000 },
				new InputModel { YearsOfExperience = 10, Salary = 83000 },
				new InputModel { YearsOfExperience = 10.3F, Salary = 86000 },
				new InputModel { YearsOfExperience = 10.5F, Salary = 90000 },

				new InputModel { YearsOfExperience = 11, Salary = 93000 },
				new InputModel { YearsOfExperience = 11.5F, Salary = 96000 },
				new InputModel { YearsOfExperience = 11.9F, Salary = 100000 },
				new InputModel { YearsOfExperience = 12, Salary = 103000 },
				new InputModel { YearsOfExperience = 12.2F, Salary = 106000 },
				new InputModel { YearsOfExperience = 12.5F, Salary = 110000 },
				new InputModel { YearsOfExperience = 13.3F, Salary = 113000 },
				new InputModel { YearsOfExperience = 13.7F, Salary = 117000 },
				new InputModel { YearsOfExperience = 13.9F, Salary = 120000 },
				new InputModel { YearsOfExperience = 14, Salary = 124000 },
				new InputModel { YearsOfExperience = 14.2F, Salary = 128000 },
				new InputModel { YearsOfExperience = 14.5F, Salary = 132000 },
				new InputModel { YearsOfExperience = 15, Salary = 136000 },
				new InputModel { YearsOfExperience = 15, Salary = 140000 },
				new InputModel { YearsOfExperience = 15, Salary = 145000 },

				new InputModel { YearsOfExperience = 16, Salary = 149000 },
				new InputModel { YearsOfExperience = 16, Salary = 153000 },
				new InputModel { YearsOfExperience = 16, Salary = 158000 },
				new InputModel { YearsOfExperience = 17, Salary = 162000 },
				new InputModel { YearsOfExperience = 17, Salary = 167000 },
				new InputModel { YearsOfExperience = 17, Salary = 172000 },
				new InputModel { YearsOfExperience = 18, Salary = 177000 },
				new InputModel { YearsOfExperience = 18, Salary = 182000 },
				new InputModel { YearsOfExperience = 18, Salary = 187000 },
				new InputModel { YearsOfExperience = 19, Salary = 192000 },
				new InputModel { YearsOfExperience = 19, Salary = 197000 },
				new InputModel { YearsOfExperience = 19, Salary = 202000 },
				new InputModel { YearsOfExperience = 20, Salary = 208000 },
				new InputModel { YearsOfExperience = 20, Salary = 214000 },
				new InputModel { YearsOfExperience = 20, Salary = 220000 },

			};

			IDataView traningData = context.Data.LoadFromEnumerable(samples);

			var estimator = context.Transforms.Concatenate("Features", nameof(InputModel.YearsOfExperience));

			var pipeline = estimator.Append(context.Regression.Trainers.Sdca(nameof(InputModel.Salary), maximumNumberOfIterations:100));

			var model = pipeline.Fit(traningData);

			var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

			var experience = new InputModel { YearsOfExperience = 5 };

			var result = predictionEngine.Predict(experience);

			Console.WriteLine($"Approx Salary for {experience.YearsOfExperience} years of experience will be : {result.Salary}");

		}
	}
}
