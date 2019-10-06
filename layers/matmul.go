package layers

import (
	"github.com/po3rin/gonnp/params"
	"gonum.org/v1/gonum/mat"
)

// MatMul layers perform the linear transformation
type MatMul struct {
	X     mat.Matrix
	Param params.Param
	Grad  params.Grad
}

// InitMatMulLayer inits matmul layer.
func InitMatMulLayer(weight mat.Matrix) *MatMul {
	return &MatMul{
		Param: params.Param{
			Weight: weight,
		},
	}
}

// Forward for matmul layer.
func (m *MatMul) Forward(x mat.Matrix) mat.Matrix {
	m.X = x
	var b mat.Dense
	b.Product(x, m.Param.Weight)
	return &b
}

// Backward for matmul layer.
func (m *MatMul) Backward(x mat.Matrix) mat.Matrix {
	var dw mat.Dense
	var dx mat.Dense

	dx.Product(x, m.Param.Weight.T())
	dw.Product(m.X.T(), x)

	m.Grad.Weight = &dw
	return &dx
}

// GetParam gets param.
func (m *MatMul) GetParam() params.Param {
	return m.Param
}

// GetGrad gets gradient.
func (m *MatMul) GetGrad() params.Grad {
	return m.Grad
}

func (m *MatMul) SetParam(p params.Param) {
	m.Param = p
}
