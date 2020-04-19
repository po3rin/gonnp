// Package xmodels has some of neural netwark models.
package xmodels

import (
	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/params"
	"gonum.org/v1/gonum/mat"
)

var (
	weightGenerator = matutil.NewRandMatrixWithSND
	biasGenerator   = matutil.NewRandVecWithSND
)

type Layer interface {
	Forward(out chan<- mat.Matrix, in ...<-chan mat.Matrix)
	Backward(out chan<- mat.Matrix, in ...<-chan mat.Matrix)
	params.Manager
}

type LossLayerWithParams interface {
	Forward(out chan<- float64, in ...<-chan mat.Matrix)
	Backward(out chan<- mat.Matrix)
	params.SetManager
}
