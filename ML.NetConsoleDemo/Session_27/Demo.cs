using Microsoft.ML;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_27
{
	internal class Demo
	{
		public static void Execute()
		{
			MLContext context = new();


			string dataPath = Path.Combine(AppContext.BaseDirectory,
							"Session_27",
							"Flight-Delay-train-dataset.csv"
						);

			var dataView = context.Data.LoadFromTextFile<InputModel>(path: dataPath, hasHeader: true, separatorChar: ',');

			var pipeline = context.Transforms.SelectColumns(nameof(InputModel.Origin), nameof(InputModel.Destination), nameof(InputModel.DepartureTime),
															nameof(InputModel.ExpectedArrivalTime), nameof(InputModel.IsDelayBy15Minutes))
							.Append(context.Transforms.Categorical.OneHotEncoding("Encoded_ORIGIN", nameof(InputModel.Origin))
							.Append(context.Transforms.Categorical.OneHotEncoding("Encoded_DESTINATION", nameof(InputModel.Destination))
							.Append(context.Transforms.DropColumns(nameof(InputModel.Origin), nameof(InputModel.Destination)))
							.Append(context.Transforms.Concatenate("Features", "Encoded_ORIGIN", "Encoded_DESTINATION", nameof(InputModel.DepartureTime), nameof(InputModel.ExpectedArrivalTime)))
							.Append(context.Transforms.Conversion.ConvertType("Label", nameof(InputModel.IsDelayBy15Minutes), Microsoft.ML.Data.DataKind.Boolean))
							.Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())
							));

			var model = pipeline.Fit(dataView);
			var preview = model.Transform(dataView).Preview();

			var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

			var input = new InputModel()
			{
				Origin = "SFO",
				Destination = "ORD",
				DepartureTime = 1300,
				ExpectedArrivalTime = 1900
			};
			PrintResult(predictionEngine.Predict(input));

			input = new InputModel()
			{
				Origin = "LAX",
				Destination = "JFK",
				DepartureTime = 900,
				ExpectedArrivalTime = 1700
			};
			PrintResult(predictionEngine.Predict(input));
		}

		static void PrintResult(ResultModel result)
		{
			Console.WriteLine($"Prediction: {result.WillDelayBy15Minutes} | Score: {result.Score}");
		}
	}
}
