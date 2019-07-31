package nn

import (
	"github.com/po3rin/gonlp/entity"
	"gonum.org/v1/gonum/mat"
)

// NeuralNet is neural network interface.
type NeuralNet interface {
	Predict(x mat.Matrix) mat.Matrix
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward() mat.Matrix
	entity.ParamsManager
}
