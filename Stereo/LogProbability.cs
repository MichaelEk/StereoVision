using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ProbabilityMethods
{
	struct LogProbability
	{
		private double _logProbability;

		public static readonly LogProbability Zero = LogProbability.GetLogProbability(0);
		public static readonly LogProbability One = LogProbability.GetLogProbability(1);
		private LogProbability(double logProbability)
		{
			_logProbability = logProbability;
		}

		private static LogProbability GetLogProbability(double probability)
		{
			return new LogProbability(Math.Log(probability));
		}

		public static LogProbability From(double probability)
		{
			return GetLogProbability(probability);
		}

		public static LogProbability FromLogProbability(double logProbability)
		{
			return new LogProbability(logProbability);
		}

		public double GetProbability()
		{
			return Math.Exp(_logProbability);
		}

		public double GetLogProbabilityValue()
		{
			return _logProbability;
		}

		public static LogProbability Max(LogProbability prob1, LogProbability prob2)
		{
			return prob1 > prob2 ? prob1 : prob2;
		}


		public static bool operator >(LogProbability prob1, LogProbability prob2)
		{
			return prob1._logProbability > prob2._logProbability;
		}

		public static bool operator <(LogProbability prob1, LogProbability prob2)
		{
			return prob1._logProbability < prob2._logProbability;
		}
		
		public static bool operator ==(LogProbability prob1, LogProbability prob2)
		{
			return Math.Abs(prob1._logProbability - prob2._logProbability) < 0.0001;
		}

		public static bool operator !=(LogProbability prob1, LogProbability prob2)
		{
			return !(prob1 == prob2);
		}

		public static LogProbability operator ~(LogProbability prob)
		{
			return new LogProbability(Math.Log(1-prob.GetProbability()));
		}

		public static LogProbability operator *(LogProbability prob1, LogProbability prob2)
		{
			return new LogProbability(prob1._logProbability + prob2._logProbability);
		}

		public static LogProbability operator *(LogProbability prob1, double prob2)
		{
			return prob1 * GetLogProbability(prob2);
		}
		public static LogProbability operator *(double prob1, LogProbability prob2)
		{
			return prob2 * prob1;
		}

		public static LogProbability operator /(LogProbability prob1, LogProbability prob2)
		{
			return new LogProbability(prob1._logProbability - prob2._logProbability);
		}

		public static LogProbability operator /(LogProbability prob1, double prob2)
		{
			return prob1/GetLogProbability(prob2);
		}

		public static LogProbability operator +(LogProbability prob1, LogProbability prob2)
		{
			if (prob1._logProbability < prob2._logProbability)
			{
				var temp = prob2;
				prob2 = prob1;
				prob1 = temp;
			}
			if (double.IsNegativeInfinity(prob1._logProbability))
				return prob1;
			return new LogProbability(prob1._logProbability + Math.Log(1 + Math.Exp(prob2._logProbability - prob1._logProbability)));
		}

		public static LogProbability Sum(IList<LogProbability> probs)
		{
			if (probs.Count == 0)
				return Zero;

			var maxLogProb = probs.Select(prob => prob._logProbability).Max();

			double sum = 0;

			for (int i = 0; i < probs.Count; i++)
			{
				sum += Math.Exp(probs[i]._logProbability - maxLogProb);
			}
			return new LogProbability(maxLogProb + Math.Log(sum));
		}

		public LogProbability Pow(double num)
		{
			return new LogProbability(num < 1e-50 ? 0 : num *_logProbability);
		}

		public override string ToString()
		{
			return GetProbability().ToString();
		}
	}

	static class LogIEnumerableExtension
	{
		public static LogProbability Sum(this IEnumerable<LogProbability> probs)
		{
			return probs.Aggregate(LogProbability.Zero, (aggr, prob) => aggr + prob);
		}

		public static LogProbability Sum(this IList<LogProbability> probs)
		{
			return LogProbability.Sum(probs);
		}

		public static LogProbability Prod(this IEnumerable<LogProbability> probs)
		{
			return probs.Aggregate(LogProbability.One, (aggr, prob) => aggr*prob);
		}

		public static LogProbability Max(this IEnumerable<LogProbability> probs)
		{
			return probs.Aggregate(LogProbability.Zero, (aggr, prob) => aggr < prob ? prob : aggr);
		}
	}
}
