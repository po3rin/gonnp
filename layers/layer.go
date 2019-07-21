package layers

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Forward(x mat.Matrix) mat.Matrix
	Backward(x, dout mat.Matrix) mat.Matrix
}

type LossLayer interface {
	Forward(x mat.Matrix) float64
}
