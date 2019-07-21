package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct {
	Output mat.Matrix
}

// InitSigmoidLayer inits sigmoid layer.
func InitSigmoidLayer() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x mat.Matrix) mat.Matrix {
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	var result mat.Dense
	result.Apply(sigmoid, x)
	s.Output = &result
	return &result
}

func (s *Sigmoid) Backward(x, dout mat.Matrix) mat.Matrix {
	backward := func(i, j int, v float64) float64 {
		return dout.At(i, j) * (1 - v) * v
	}
	var result mat.Dense
	result.Apply(backward, s.Output)
	return &result
}
