using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_27
{
	internal class ResultModel
	{
		[ColumnName("PredictedLabel")]
		public bool WillDelayBy15Minutes { get; set; }

		public float Score { get; set; }
	}
}
