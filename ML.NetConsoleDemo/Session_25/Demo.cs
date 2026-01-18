using Microsoft.ML;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_25
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new();

			var filePath = Path.Combine(AppContext.BaseDirectory, "Session_16", "salary_data_high_accuracy.csv");

			IDataView dataView = context.Data.LoadFromTextFile<InputModel>(path: filePath, hasHeader: true, separatorChar: ',');

			var estimator = context.Transforms.Concatenate("Features", nameof(InputModel.YearsOfExperience));

			var pipeline = estimator.Append(context.Regression.Trainers.Sdca(labelColumnName: nameof(InputModel.Salary), maximumNumberOfIterations: 100));

			var model = pipeline.Fit(dataView);

			string modelPath = Path.Combine(AppContext.BaseDirectory,
							"Session_24",
							"salary-prediction-model.zip"
						);

			var directory = Path.GetDirectoryName(modelPath);
			if (!Directory.Exists(directory))
				Directory.CreateDirectory(directory!);

			if(!File.Exists(modelPath))
				context.Model.Save(model, dataView.Schema, Path.Combine(AppContext.BaseDirectory, modelPath));

			var saveModel = context.Model.Load(modelPath, out DataViewSchema schema);

			var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(saveModel);

			var experience = new InputModel() { YearsOfExperience = 7.5F };

			var prediction = predictionEngine.Predict(experience);

			Console.WriteLine($"Predicted Salary for {experience.YearsOfExperience} years of experience is {prediction.Salary:C2}");
		}
	}
}
