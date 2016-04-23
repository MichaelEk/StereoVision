using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace EMText
{
	class SemGaussianMixtureLearner
	{
		static Random rand = new Random();

		public static void DoEMStep(List<MultivariateGaussian> gaussians, double[] weights, List<double[]> vectors)
		{
			var probsVectorsByGaussians = new double[vectors.Count][];
			var newWeights = new double[gaussians.Count];

			Enumerable.Range(0, vectors.Count)
				.AsParallel()
				.ForAll(i =>
				{
					probsVectorsByGaussians[i] = new double[gaussians.Count];
					for (int j = 0; j < gaussians.Count; j++)
					{
						probsVectorsByGaussians[i][j] = gaussians[j].GetDensity(vectors[i]) * weights[j];
					}
					var norm = probsVectorsByGaussians[i].Sum();
					if (norm < 1e-100)
						return;
					probsVectorsByGaussians[i] = probsVectorsByGaussians[i].Divide(norm);

					lock (newWeights)
					{
						for (int j = 0; j < gaussians.Count; j++)
							newWeights[j] += probsVectorsByGaussians[i][j];
					}
				});

			Array.Copy(newWeights.Divide(newWeights.Sum()), weights, weights.Length);

			var idxs = probsVectorsByGaussians.Select(GetIndex).ToList();

			Enumerable.Range(0, gaussians.Count)
				.AsParallel()
				.ForAll(j => gaussians[j].Learn(vectors.Where((vector, i) => idxs[i] == j).ToList()));
		}

		private static int GetIndex(double[] weights)
		{
			var w = rand.NextDouble()*weights.Sum();
			int res = 0;
			while (weights[res] < w)
			{
				w -= weights[res];
				res++;
			}
			return res;
		}
	}
}
