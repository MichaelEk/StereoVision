using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Accord.Math;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using ProbabilityMethods;

namespace EMText
{
	public partial class Form1 : Form
	{
		private MarkovNetwork Network;

		private Image<Rgb, Byte> Left;
		private Image<Rgb, Byte> Right;
		private Image<Rgb, Byte> Image;

		public Form1()
		{
			InitializeComponent();
			Left = new Image<Rgb, byte>("4right.png").Resize(320, 240, INTER.CV_INTER_NN);
			Right = new Image<Rgb, byte>("4left.png").Resize(320, 240, INTER.CV_INTER_NN);
			Image = new Image<Rgb, byte>(Left.Width, Left.Height);
			ImageBox1.Image = Left.PyrUp();
			Update();
			Network = new MarkovNetwork(Left, Right);
			UpdateImage();
		}


		private int iterationsToLearnWithoutSuccess;
		private LogProbability Max;

		private int LearnTimesForIteration = 10;

		private void UpdateImage()
		{
			Image = Image.AddWeighted(Network.GetDeepImage(), 0, 1, 1);
            ImageBox2.Image = Image;
			ImageBox1.Image = Network.GetMixImage(Right.Clone());
			Update();
		}

		private int Idx = 2;

		private void button1_Click(object sender, EventArgs e)
		{
			Idx = (Idx + 1)%9;

			Left = new Image<Rgb, byte>((Idx + 1) + "right.png").Resize(320, 240, INTER.CV_INTER_NN);
			Right = new Image<Rgb, byte>((Idx + 1) + "left.png").Resize(320, 240, INTER.CV_INTER_NN);
			Network = new MarkovNetwork(Left, Right);
			
			while (Network.Step > 1)
			{
				LogProbability maxProb = LogProbability.Zero;
				int afterMaxStepsCount = 0;

				while(afterMaxStepsCount < 5)
				{
					afterMaxStepsCount++;
					Network.HiggsStep();
					UpdateImage();
					var prob = Network.GetPotential();
					if (prob > maxProb)
					{
						maxProb = prob;
						afterMaxStepsCount = 0;
					}
				} 

				Network.ReduceStep();
			}

			for (int i = 0; i < 100; i++)
			{
				Network.HiggsStep();
				UpdateImage();
			}
		}

		private void button2_Click(object sender, EventArgs e)
		{
			Network.ReduceStep();
		}

		private void Form1_FormClosing(object sender, FormClosingEventArgs e)
		{
		}
	}
}
