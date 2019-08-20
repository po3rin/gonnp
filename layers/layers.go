// Package layers impliments various layer for neural network.
package layers

import (
	"github.com/po3rin/gonnp/entity"
	"gonum.org/v1/gonum/mat"
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
