package layers

import (
	"math"

	"github.com/po3rin/gonlp/functions"
	"gonum.org/v1/gonum/mat"
)

// SoftmaxWithLoss is layer for computing the multinomial logistic loss of the softmax of its inputs
type SoftmaxWithLoss struct {
	Output  mat.Matrix
	Teacher mat.Matrix
}

// InitSoftmaxWithLossLayer inits softmax layer.
func InitSoftmaxWithLossLayer() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

// Forward for softmax layer.
func (s *SoftmaxWithLoss) Forward(x mat.Matrix, teacher mat.Matrix) float64 {
	s.Output = Softmax(x)
	s.Teacher = teacher
	return functions.CrossEntropyErr(s.Output, teacher)
}

// Backward for softmax layer.
func (s *SoftmaxWithLoss) Backward(x mat.Matrix) float64 {
	panic("no impliments")
}

func Softmax(x mat.Matrix) mat.Matrix {
	c := mat.Max(x)
	f := func(i, j int, v float64) float64 {
		return math.Exp(v - c)
	}

	var result mat.Dense
	result.Apply(f, x)

	sum := mat.Sum(&result)

	result.Scale(1/sum, &result)
	return &result
}
