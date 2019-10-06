// Package models has some of neural netwark models.
package models

import (
	"github.com/po3rin/gonnp/matutils"
	"github.com/po3rin/gonnp/params"
	"gonum.org/v1/gonum/mat"
)

var (
	weightGenerator = matutils.NewRandMatrixWithSND
	biasGenerator   = matutils.NewRandVecWithSND
)

type Layer interface {
	Forward(x mat.Matrix) mat.Matrix
	Backward(x mat.Matrix) mat.Matrix
	params.Manager
}

type LossLayer interface {
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward() mat.Matrix
}

type LossLayerWithParams interface {
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward() mat.Matrix
	params.SetManager
}
