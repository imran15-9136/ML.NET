using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_10
{
	internal class InputModel
	{
		[LoadColumn(0)]
		public float YearsOfExperience { get; set; }
		[LoadColumn(1)]
		public float Salary { get; set; }
	}
}
