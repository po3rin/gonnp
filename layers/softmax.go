package layers

import (
	"math"

	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

// SoftmaxWithLoss is layer for computing the multinomial logistic loss of the softmax of its inputs
type SoftmaxWithLoss struct {
	X       mat.Matrix
	Teacher mat.Matrix
}

// InitSoftmaxWithLossLayer inits softmax layer.
func InitSoftmaxWithLossLayer() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

// Forward for softmax layer.
func (s *SoftmaxWithLoss) Forward(x mat.Matrix, teacher mat.Matrix) float64 {
	s.X = softmax(x)
	s.Teacher = teacher
	return crossEntropyErr(s.X, teacher)
}

// Backward for softmax layer.
func (s *SoftmaxWithLoss) Backward() mat.Matrix {
	batchSize, c := s.Teacher.Dims()
	f := func(i, j int, v float64) float64 {
		return (v - s.Teacher.At(i, j)) / float64(batchSize)
	}
	dx := mat.NewDense(batchSize, c, nil)
	dx.Apply(f, s.X)
	return dx
}

func softmax(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	if r == 1 || c == 1 {
		max := mat.Max(x)
		f := func(i, j int, v float64) float64 {
			return math.Exp(v - max)
		}
		var result mat.Dense
		result.Apply(f, x)
		sum := mat.Sum(&result)
		result.Scale(1/sum, &result)
		return &result
	}
	d := mat.NewDense(r, c, nil)
	m := matutils.SubMatVec(x, matutils.Mat2VecWithColMax(x))
	f := func(i, j int, v float64) float64 {
		return math.Exp(v)
	}
	d.Apply(f, m)
	v := matutils.SumRow(d)
	result := matutils.DivMatVec(d, v)
	return result
}

// CrossEntropyErr measures the performance of a classification model whose output is a probability value between 0 and 1.
func crossEntropyErr(data mat.Matrix, teacher mat.Matrix) float64 {
	batchSize, _ := data.Dims()
	tr, tc := teacher.Dims()

	delta := 1e-7
	crossEnrtopy := func(i, j int, v float64) float64 {
		return -1 * v * math.Log(data.At(i, j)+delta)
	}

	if batchSize == 1 {
		ce := mat.NewDense(tr, tc, nil)
		ce.Apply(crossEnrtopy, teacher)
		return mat.Sum(ce)
	}

	ce := mat.NewDense(tr, tc, nil)
	ce.Apply(crossEnrtopy, teacher)
	return mat.Sum(ce) / float64(batchSize)
}
