package layers

import "gonum.org/v1/gonum/mat"

type MutMul struct {
	X      mat.Matrix
	Weight mat.Matrix
	Grads  mat.Matrix
}

func (m *MutMul) Forward(x, dout mat.Matrix) mat.Matrix {
	return x
}
