using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Extensions.Math;
using Accord.Extensions.Math.Geometry;
using Accord.Math;
using Accord.Math.Decompositions;
using Accord.Statistics.Kernels;
using AForge.Math;

namespace EMText
{
	class Gaussian
	{
		public double Mean;
		public double Variance;

		public Gaussian()
		{
			Mean = 0;
			Variance = 1;
		}

		public Gaussian(double mean, double variance)
		{
			Mean = mean;
			Variance = variance;
		}

		public void Learn(List<double> values)
		{
			Mean = values[0];
			for (int i = 1; i < values.Count; i++)
				Mean = (Mean*i + values[i])/(i+1);

			Variance = values.Select(value => Math.Pow(value - Mean, 2)).Average();
		}

		public double GetDensity(double x)
		{
			return 1/Math.Sqrt(2*Math.PI*Variance)*Math.Exp(-Math.Pow(x - Mean, 2)/2/Variance);
		}

		private double GetCDF(double x)
		{
			return 0.5 + LaplaceFunction.From((x - Mean)/Math.Sqrt(Variance));
		}

		private static Random rand = new Random();

		public double GenerateRandom()
		{
			return LaplaceFunction.FromValue(rand.NextDouble() - 0.5) * Math.Sqrt(Variance) + Mean;
		}
	}

	class MultivariateGaussian
	{
		public double[] Mean;
		public double[,] CovarianceMatrix;

		private double[,] EigenMatrix;

		private bool Learned = false;
		private double UnlearnedDefaultDensity = 1e-10;

		public double Max;
		
		public MultivariateGaussian(int dimSize, double unlearnedDefaultDensity = 1e-10)
		{
			Mean = new double[dimSize];
			CovarianceMatrix = new double[dimSize, dimSize];
			for (int i = 0; i < dimSize; i++)
			{
				CovarianceMatrix[i, i] = 1;
			}
			UnlearnedDefaultDensity = unlearnedDefaultDensity;
		}

		public MultivariateGaussian(double[] mean, double[,] covarianceMatrix)
		{
			Mean = mean;
			CovarianceMatrix = covarianceMatrix;
			EigenMatrix = covarianceMatrix.GetEigenMatrix();
			Learned = true;

			Max = GetDensity(mean.Select(v => 0.0).ToArray());
		}

		public void Learn(List<double[]> vectors, List<double> weights)
		{
			if(vectors.Count <= 2)
				return;

			Mean = new double[Mean.Length];

			for (int index = 0; index < vectors.Count; index++)
			{
				var vector = vectors[index];
				Mean = Mean.Add(vector.Multiply(weights[index]));
			}

			Mean = Mean.Divide(weights.Sum());

			CovarianceMatrix = new double[vectors[0].Length, vectors[0].Length];

			for (int index = 0; index < vectors.Count; index++)
			{
				var vector = vectors[index];
				var diffVector = vector.Subtract(Mean).ToMatrix();
				var covMatirix = diffVector.Transpose().Multiply(diffVector);
				CovarianceMatrix = CovarianceMatrix.Add(covMatirix.Multiply(weights[index]));
			}

			CovarianceMatrix = CovarianceMatrix.Divide(weights.Sum());

			EigenMatrix = CovarianceMatrix.GetEigenMatrix();

			Learned = true;
		}

		public void Learn(List<double[]> vectors)
		{
			Learn(vectors, vectors.Select(item => 1.0).ToList());
		}

		public double GetDensity(double[] vector)
		{
			if (!Learned)
				return UnlearnedDefaultDensity;

			var deltaVector = vector.Subtract(Mean);
			if (CovarianceMatrix.Rank() < vector.Length)
				return 0;

			return 1/Math.Sqrt(Math.Pow(2*Math.PI, vector.Length)*CovarianceMatrix.Determinant())*
			       Math.Exp(-0.5*deltaVector.Multiply(CovarianceMatrix.Inverse()).Multiply(deltaVector.Transpose())[0]);
		}

		private static Gaussian NormalGaussian = new Gaussian(0, 1);

		private static Random rand = new Random();

		public double[] GenerateRandom()
		{
			if (!Learned)
				return new[] {rand.NextDouble()*255, rand.NextDouble()*255, rand.NextDouble()*255};

			var result = new double[Mean.Length];

			for (int i = 0; i < result.Length; i++)
			{
				result[i] = NormalGaussian.GenerateRandom();
			}
			
			return EigenMatrix.Multiply(result).Add(Mean);
		}
	}

	public static class MatrixHelper
	{
		public static double[,] GetEigenMatrix(this double[,] matrix)
		{
			var dec = new EigenvalueDecomposition(matrix);

			return dec.Eigenvectors.Multiply(dec.RealEigenvalues.Select(Math.Sqrt).ToArray().ToDiagonalMatrix());
		}
	}
}
