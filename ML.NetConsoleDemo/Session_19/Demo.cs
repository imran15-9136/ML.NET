using Microsoft.ML;
using Microsoft.ML.Data;
using ML.NetConsoleDemo.Session_10;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetConsoleDemo.Session_19
{
	internal class Demo
	{
		public static void Execute() 
		{
			MLContext context = new();

			var database = context.Data.CreateDatabaseLoader<InputModel>();

			var connectionString = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=MLNetDemoDB;Integrated Security=True;Connect Timeout=30;Encrypt=False;TrustServerCertificate=False;ApplicationIntent=ReadWrite;MultiSubnetFailover=False";

			string commandText = "SELECT YearsOfExperience, Salary FROM dbo.SalaryData";

			var dataSourse = new DatabaseSource(SqlClientFactory.Instance, connectionString, commandText);

			IDataView dataView = database.Load(dataSourse);

			var preview = dataView.Preview();

		}
	}
}
