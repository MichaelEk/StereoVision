using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace EMText
{
	class EMGaussMixture
	{
		public static void DoEMStep(List<MultivariateGaussian> gaussians, double[] weights, List<double[]> vectors)
		{
			var probsVectorsByGaussians = new double[vectors.Count][];
			var newWeights = new double[gaussians.Count];

			for (int i = 0; i < vectors.Count; i++)
			{
				probsVectorsByGaussians[i] = new double[gaussians.Count];
				for (int j = 0; j < gaussians.Count; j++)
				{
					probsVectorsByGaussians[i][j] = gaussians[j].GetDensity(vectors[i])*weights[j];
				}
				probsVectorsByGaussians[i] = probsVectorsByGaussians[i].Divide(probsVectorsByGaussians[i].Sum());
				for (int j = 0; j < gaussians.Count; j++)
					newWeights[j] += probsVectorsByGaussians[i][j];
			}

			Array.Copy(newWeights.Divide(newWeights.Sum()), weights, weights.Length);

			for (int j = 0; j < gaussians.Count; j++)
			{
				gaussians[j].Learn(vectors, Enumerable.Range(0, vectors.Count).Select(i => probsVectorsByGaussians[i][j]).ToList());
			}
		}
	}
}
