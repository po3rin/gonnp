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

type Layer interface {
	Forward(x mat.Matrix) mat.Matrix
	Backward(x mat.Matrix) mat.Matrix
	entity.ParamManager
}

type LossLayer interface {
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward() mat.Matrix
}

type LossLayerWithParams interface {
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward() mat.Matrix
	entity.ParamsManager
}
