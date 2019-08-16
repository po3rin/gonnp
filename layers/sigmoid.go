package layers

import (
	"math"

	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/matutils"
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
	s.X = sigmoid(x)
	return s.X
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

// SigmoidWithLoss is layer for computing the multinomial logistic loss of the sigmoid of its inputs
type SigmoidWithLoss struct {
	X       mat.Matrix
	Teacher mat.Matrix
}

// InitSigmoidWithLossLayer inits sigmoid loss layer.
func InitSigmoidWithLossLayer() *SigmoidWithLoss {
	return &SigmoidWithLoss{}
}

// Forward for sigmoid loss layer.
func (s *SigmoidWithLoss) Forward(x mat.Matrix, teacher mat.Matrix) float64 {
	x = sigmoid(x)
	s.X = x
	s.Teacher = teacher

	y := mat.DenseCopyOf(x)
	f := func(i, j int, v float64) float64 {
		return 1 - v
	}
	y.Apply(f, y)
	return crossEntropyErr(matutils.JoinC(y, x), teacher)
}

// Backward for sigmoid loss layer.
func (s *SigmoidWithLoss) Backward() mat.Matrix {
	batchSize, c := s.Teacher.Dims()
	f := func(i, j int, v float64) float64 {
		return (v - s.Teacher.At(i, j)) / float64(batchSize)
	}
	dx := mat.NewDense(batchSize, c, nil)
	dx.Apply(f, s.X)
	return dx
}

func sigmoid(x mat.Matrix) mat.Matrix {
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}

	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(sigmoid, x)
	return result
}
