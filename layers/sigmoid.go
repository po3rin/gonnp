package layers

import (
	"math"

	"github.com/po3rin/gonnp/entity"
	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct {
	X     mat.Matrix
	Param entity.Param
	Grad  entity.Grad
}

// InitSigmoidLayer inits sigmoid layer.
func InitSigmoidLayer() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x mat.Matrix) mat.Matrix {
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}

	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(sigmoid, x)
	s.X = result
	return result
}

func (s *Sigmoid) Backward(x mat.Matrix) mat.Matrix {
	backward := func(i, j int, v float64) float64 {
		return x.At(i, j) * (1 - v) * v
	}

	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(backward, s.X)
	return result
}

func (s *Sigmoid) GetParam() entity.Param {
	return s.Param
}

func (s *Sigmoid) GetGrad() entity.Grad {
	return s.Grad
}

func (s *Sigmoid) SetParam(p entity.Param) {
	s.Param = p
}
