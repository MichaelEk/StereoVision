using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace EMText
{
	class GaussiansMixture
	{
		private static Random rand = new Random();

		public List<MultivariateGaussian> Gaussians = new List<MultivariateGaussian>();
		public double[] Weights = new double[0];

		public void Init()
		{
			for (int i = 0; i < 3; i++)
			{
				AddGaussian();
			}
		}
		
		public void Learn(List<double[]> vectors)
		{
			SemGaussianMixtureLearner.DoEMStep(Gaussians, Weights, vectors);
			var gaussiansToDelete = Enumerable.Range(0, Gaussians.Count)
				.Where(i => Weights[i] < 0.005)
				.Reverse();

			foreach (var idx in gaussiansToDelete)
			{
				RemoveGaussian(idx);
			}
		}

		public double[] GetMaxMean(double[] vector, out double max)
		{
			double[] maxMean = null;
			max = 0;

			for (int i = 0; i < Gaussians.Count; i++)
			{
				var weight = Gaussians[i].GetDensity(vector)*Weights[i];
				if (weight > max)
				{
					maxMean = Gaussians[i].Mean;
					max = weight;
				}
			}
			
			return maxMean ?? vector.Select(v => 0.0).ToArray();
		}

		public double[] Generate(double[] vector, out double max)
		{
			MultivariateGaussian gaussian = null;
			max = 0;

			for (int i = 0; i < Gaussians.Count; i++)
			{
				var weight = Gaussians[i].GetDensity(vector) * Weights[i];
				if (weight > max)
				{
					gaussian = Gaussians[i];
					max = weight;
				}
			}

			return gaussian?.GenerateRandom() ?? new [] {0.0, 0, 0};
		}

		private void RemoveGaussian(int idx)
		{
			Gaussians.RemoveAt(idx);
			Weights = Weights.Take(idx).Concat(Weights.Skip(idx + 1)).ToArray();
		}

		public void AddGaussian()
		{
			Gaussians.Add(new MultivariateGaussian(3, Math.Pow(1.0/255, 3)));
			Weights = Weights.Concat(new[] { 0.5 / Gaussians.Count }).ToArray();
			Weights = Weights.Divide(Weights.Sum());
		}
	}
}
