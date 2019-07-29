package layers

import (
	"github.com/po3rin/gonlp/entity"
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Forward(x mat.Matrix) mat.Matrix
	Backward(x mat.Matrix) mat.Matrix
	GetParamAndGrad() (entity.Param, entity.Grad)
}

type LossLayer interface {
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward(x mat.Matrix) mat.Matrix
}
