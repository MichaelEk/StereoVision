using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Accord.Math;
using AForge.Math;
using Emgu.CV;
using Emgu.CV.Structure;
using ProbabilityMethods;

namespace EMText
{
	class MarkovNetwork
	{
		public MarkovNode[,] Net;
		private List<MarkovNode> Nodes = new List<MarkovNode>();
		private double[,][] RightMap;
		

		public MarkovNetwork(Image<Rgb, byte> left, Image<Rgb, byte> right)
		{
			RightMap = new double[right.Height, right.Width][];
			for (int y = 0; y < right.Height; y++)
			{
				for (int x = 0; x < right.Width; x++)
				{
					var pixel = right[y, x];
					RightMap[y, x] = new[] {pixel.Red, pixel.Green, pixel.Blue};
				}
			}

			Net = new MarkovNode[left.Height, left.Width];

			for (int y = 0; y < left.Height; y++)
			{
				for (int x = 0; x < left.Width; x++)
				{
					var pixel = left[y, x];
					Net[y, x] = new MarkovNode(x, y, new[] { pixel.Red, pixel.Green, pixel.Blue }, RightMap);
					Nodes.Add(Net[y, x]);
				}
			}

			SetEdges();

		}

		public int Step = 16;

		public void ReduceStep()
		{
			Step /= 2;
			SetEdges();
			Resize();
		}

		private void SetEdges()
		{
			/*
			for (int y = 0; y < Net.GetLength(0); y++)
			{
				for (int x = 0; x < Net.GetLength(1); x++)
				{
					Net[y, x].Edges.Clear();
                }
			}
			*/

			for (int y = 0; y < Net.GetLength(0); y++)
			{
				for (int x = 0; x < Net.GetLength(1); x++)
				{
					if (x >= Step)
					{
						Net[y, x].Edges.Add(Net[y, x - Step]);
						Net[y, x - Step].Edges.Add(Net[y, x]);
					}
					if (y >= Step)
					{
						Net[y, x].Edges.Add(Net[y - Step, x]);
						Net[y - Step, x].Edges.Add(Net[y, x]);
					}
				}
			}

			for (int y = 0; y < Net.GetLength(0); y++)
			{
				for (int x = 0; x < Net.GetLength(1); x++)
				{
					Net[y, x].Init();
				}
			}
		}

		private void Resize()
		{
			for (int y = 0; y < Net.GetLength(0); y += Step*2)
			{
				for (int x = 0; x < Net.GetLength(1); x += Step*2)
				{
					var s = Net[y, x].Dx;//*2;
					Net[y, x].Dx = s;
					Net[y + Step, x].Dx = ChooseResizeDensity(x, y, x, y + Step * 2);
					Net[y, x + Step].Dx = ChooseResizeDensity(x, y, x + Step * 2, y);
					Net[y + Step, x + Step].Dx = ChooseResizeDensity(x, y, x + Step * 2, y + Step * 2);
				}
			}
		}

		private int ChooseResizeDensity(int x1, int y1, int x2, int y2)
		{
			if (x2 >= Net.GetLength(1) || y2 >= Net.GetLength(0))
				return Net[y1, x1].Dx;

			if(Net[y1,x1].Dx == Net[y2, x2].Dx)
				return Net[y1, x1].Dx;
			var min = Math.Min(Net[y1, x1].Dx, Net[y2, x2].Dx);
			var max = Math.Max(Net[y1, x1].Dx, Net[y2, x2].Dx);

			return min + rand.Next(max - min + 1);
		}

		private static Random rand = new Random();
		
		public void HiggsStep()
		{
			for (int i = 0; i < 2; i++)
			{
				Enumerable.Range(0, Net.GetLength(0)/Step)
					.AsParallel()
					.ForAll(yStep =>
					{
						for (int x = 0; x < Net.GetLength(1); x += Step)
						{
							if ((x / Step + yStep) % 2 == i)
								continue;

							Net[yStep*Step, x].HiggsStep();
						}
					});
			}
		}
		
		public Image<Rgb, byte> GetMixImage(Image<Rgb, byte> image)
		{
			var result = new Image<Rgb, byte>(image.Width, image.Height);
			Enumerable.Range(0, result.Height)
				.AsParallel()
				.ForAll(y =>
				{
					for (int x = result.Width - 1; x >= 0; x--)
					{
						if (Net[y - y % Step, x - x % Step].Hidden)
							continue;
						var proj = Math.Max(0, Math.Min(image.Width - 1, x + Net[y - y % Step, x - x % Step].Dx));
						var color = Net[y - y % Step, x - x % Step].Color;
						result[y, proj] = new Rgb(color[0], color[1], color[2]);
					}
				});

			return result;
		}

		public Image<Rgb, byte> GetDeepImage()
		{
			var nodes = Nodes.Where(node => !node.Hidden).OrderBy(node => node.Dx).ToList();

			var min = nodes[nodes.Count/20].Dx-1;
			var max = nodes[nodes.Count - nodes.Count / 20].Dx;

			var result = new Image<Rgb, byte>(Net.GetLength(1), Net.GetLength(0));

			Enumerable.Range(0, result.Height)
				.AsParallel()
				.ForAll(y =>
				{
					for (int x = 0; x < result.Width; x++)
					{
						var vector = Net[y - y % Step, x - x % Step].Hidden ? 0 : Math.Max(Math.Min((Net[y - y % Step, x - x % Step].Dx - min) * 255.0 / (max - min), 255), 0);
						result[y, x] = new Rgb(vector, vector, vector);
					}
				});

			return result;
		}

		public LogProbability GetPotential()
		{
			LogProbability prob = LogProbability.One;

			for (int y = 0; y < RightMap.GetLength(0); y+=Step)
			{
				for (int x = 0; x < RightMap.GetLength(1); x+=Step)
				{
					prob *= Net[y, x].GetRightDownProb();
				}
			}
			return prob;
		}
	}

	class MarkovNode
	{
		private static MultivariateGaussian compareGaussian = new MultivariateGaussian(new[] { 0.0, 0, 0 }, Matrix.Identity(3).Multiply(70));
		
		public static Dictionary<int, Gaussian> nearDxGaussians = new Dictionary<int, Gaussian>();
		public double[] ColorToContinue = null;

		private static double EdgeProb = 0.01;

		public int X;
		public int Y;
		public double[] Color;
		public List<MarkovNode> Edges = new List<MarkovNode>();
		public double Prob;
		public bool Hidden = false;

		public int Dx = 0;
		
		[ThreadStatic ]private static Random rand;

		private double[,][] RightMap;
		public int[,] DensityMap; 

		public MarkovNode(int x, int y, double[] color, double[,][] rightMap)
		{
			if(rand == null)
				rand = new Random();
			X = x;
			Y = y;
			Color = color;
			Dx = rand.Next(rightMap.GetLength(1)) - x;
			RightMap = rightMap;
			Prob = 1;
		}

		public void Init()
		{
            foreach (var edge in Edges)
			{
				var diff = SqrDist(edge);
				if (!nearDxGaussians.ContainsKey(diff))
					nearDxGaussians[diff] = new Gaussian(0, 4*Math.Sqrt(diff));
			}
		}
		
		private Dictionary<int, double> Variants = new Dictionary<int, double>(); 

		public void HiggsStep()
		{
			if(rand == null)
				rand = new Random();
			
			Variants.Clear();

			var points = Edges.Select(edge => edge.Dx)
				.Concat(new[] { Dx })
				.Distinct()
				.ToList();

			foreach (var point in points)
			{
				Variants[point] = GetColorMetric(point);
			}

			foreach (var node in Edges)
			{
				var variants = node.GenerateProbabilities(points, this);
				foreach (var variant in variants)
				{
					Variants[variant.Key] *= variant.Value;
				}
			}

			ChooseRandom();

			Hidden = IsOutBorders(Dx);
		}

		private void ChooseRandom()
		{
			if (rand == null)
				rand = new Random();

			var sum = Variants.Sum(pair => pair.Value);
			var r = rand.NextDouble()*sum;

			foreach (var pair in Variants)
			{
				r -= pair.Value;
				if (r < 0)
				{
					Prob = pair.Value/sum;
					Dx = pair.Key;
					return;
				}
			}
		}


		[ThreadStatic] private static Dictionary<int, double> vars; 
		
		public Dictionary<int, double> GenerateProbabilities(IEnumerable<int> variants, MarkovNode node)
		{
			if(vars == null)
				vars = new Dictionary<int, double>();
			vars.Clear();

			foreach (var variant in variants)
			{
				vars[variant] = GetEdgeMetric(this, Dx, node, variant);
			}
			return vars;
		}

		private static int MaxDelta = 3;

		private double GetEdgeMetric(MarkovNode node1, int dx1, MarkovNode node2, int dx2)
		{
			var dist = SqrDist(node1, node2);

			return Math.Abs(dx1 - dx2) > MaxDelta
				? nearDxGaussians[dist].GetDensity(MaxDelta)
				: nearDxGaussians[dist].GetDensity(dx1 - dx2);
		}

		private int SqrDist(MarkovNode node)
		{
			return SqrDist(node, this);
		}

		private static int SqrDist(MarkovNode node1, MarkovNode node2)
		{
			return (node1.Y - node2.Y) * (node1.Y - node2.Y) + (node1.X - node2.X) * (node1.X - node2.X);
		}

		public LogProbability GetRightDownProb()
		{
			var prob = LogProbability.One;

			prob *= GetColorMetric(Dx);

			foreach (var edge in Edges.Where(edge => edge.X >= X && edge.Y >= Y))
			{
				prob *= GetEdgeMetric(this, this.Dx, edge, edge.Dx);
			}
			return prob;
		}

		private double GetColorMetric(int dx)
		{
			if (IsOutBorders(dx))
				return 1e-8;
			return GetColorMetric(Color, RightMap[Y, X + dx]);
		}

		private bool IsOutBorders(int dx)
		{
			return X + dx < 0 || X + dx >= RightMap.GetLength(1);
		}


		private double GetColorMetric(double[] color1, double[] color2)
		{
			return compareGaussian.GetDensity(color1.Subtract(color2));
		}
	}

	class DerivativeNode
	{
		public double Prob;
		public int Value;
		public MarkovNode LeftNode;
		public MarkovNode RightNode;

		public List<DerivativeNode> Edges;
	}
}
