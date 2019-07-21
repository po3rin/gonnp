package layers

import (
	"math"

	"github.com/po3rin/gonlp/functions"
	"gonum.org/v1/gonum/mat"
)

// type Softmax struct {
// 	// prepare type insted of interface.
// 	Params interface{}
// 	Grads  interface{}
// 	Output mat.Matrix
// }

// InitSoftmaxLayer inits softmax layer.
// func InitSoftmaxLayer() *Softmax {
// 	return &Softmax{}
// }

// func (s *Softmax) Forward(x mat.Matrix) mat.Matrix {
// 	s.Output = softmax(x)
// 	return s.Output
// }

type SoftmaxWithLoss struct {
	// prepare type insted of interface.
	Params  interface{}
	Grads   interface{}
	Output  mat.Matrix
	Teacher mat.Matrix
}

// InitSoftmaxWithLossLayer inits softmax layer.
func InitSoftmaxWithLossLayer() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

func (s *SoftmaxWithLoss) Forward(x mat.Matrix) float64 {
	y := softmax(x)
	return functions.CrossEntropyErr(y, s.Teacher)
}

func (s *SoftmaxWithLoss) Backward(x mat.Matrix) float64 {
	return 0
}

func softmax(x mat.Matrix) mat.Matrix {
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
