package layers

import (
	"math"

	"github.com/po3rin/gonlp/entity"
	"gonum.org/v1/gonum/mat"
)

// SoftmaxWithLoss is layer for computing the multinomial logistic loss of the softmax of its inputs
type SoftmaxWithLoss struct {
	X       mat.Matrix
	Teacher mat.Matrix
	Param   entity.Param
	Grad    entity.Grad
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
		// return (s.X.At(i, j) - s.Teacher.At(i, j)) * x.At(i, j) / float64(batchSize)
		return (v - s.Teacher.At(i, j)) / float64(batchSize)
	}
	dx := mat.NewDense(batchSize, c, nil)
	dx.Apply(f, s.X)
	return dx
}

func (s *SoftmaxWithLoss) GetParamAndGrad() (entity.Param, entity.Grad) {
	return s.Param, s.Grad
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

// CrossEntropyErr measures the performance of a classification model whose output is a probability value between 0 and 1.
func crossEntropyErr(data mat.Matrix, teacher mat.Matrix) float64 {
	// TODO: if teacher data is one-hot, ignore 0 in teacher data.
	batchSize, _ := data.Dims()
	var ce mat.Dense
	ce.Apply(crossEnrtopy, data)

	var mul mat.Dense
	mul.MulElem(teacher, &ce)

	sum := mat.Sum(&mul)
	return -sum / float64(batchSize)
}

func crossEnrtopy(i, j int, v float64) float64 {
	delta := 0.0000001
	return math.Log(v + delta)
}
