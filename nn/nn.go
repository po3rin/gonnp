// Package nn is some of neural netwark models.
package nn

import (
	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/matutils"
	"gonum.org/v1/gonum/mat"
)

var (
	weightGenerator = matutils.NewRandMatrixWithSND
	biasGenerator   = matutils.NewRandVecWithSND
	// biasGenerator = mat.NewVecDense
)

// NeuralNet is neural network interface.
type NeuralNet interface {
	Forward(teacher mat.Matrix, x ...mat.Matrix) float64
	Backward() mat.Matrix
	entity.ParamsManager
}
