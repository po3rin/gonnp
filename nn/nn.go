package nn

import (
	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

var (
	weightGenerator = matutils.NewRandMatrixWithSND
	biasGenerator   = matutils.NewRandVecWithSND
	// biasGenerator = mat.NewVecDense
)

// NeuralNet is neural network interface.
type NeuralNet interface {
	Predict(x mat.Matrix) mat.Matrix
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward() mat.Matrix
	entity.ParamsManager
}
