package nn

import "gonum.org/v1/gonum/mat"

type NeralNet interface {
	Predict(x mat.Matrix) mat.Matrix
	Forward(x mat.Matrix, teacher mat.Matrix) float64
	Backward(x mat.Matrix) mat.Matrix
}
