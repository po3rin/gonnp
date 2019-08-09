package layers

import (
	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/matutils"
	"gonum.org/v1/gonum/mat"
)

// Affine layers perform the linear transformation
type Affine struct {
	X     mat.Matrix
	Param entity.Param
	Grad  entity.Grad
}

// InitAffineLayer inits affine layer.
func InitAffineLayer(weight mat.Matrix, bias mat.Vector) *Affine {
	return &Affine{
		Param: entity.Param{
			Weight: weight,
			Bias:   bias,
		},
	}
}

// Forward for affine layer.
func (a *Affine) Forward(x mat.Matrix) mat.Matrix {
	a.X = x

	var b mat.Dense
	b.Product(x, a.Param.Weight)

	var c mat.Dense
	add := func(i, j int, v float64) float64 {
		return v + a.Param.Bias.AtVec(j)
	}
	c.Apply(add, &b)

	return &c
}

// Backward for affine layer.
func (a *Affine) Backward(x mat.Matrix) mat.Matrix {
	var dw mat.Dense
	var dx mat.Dense

	dx.Product(x, a.Param.Weight.T())
	dw.Product(a.X.T(), x)
	db := matutils.SumCol(x)

	a.Grad.Weight = &dw
	a.Grad.Bias = db

	// matutils.PrintMat(&dx)
	return &dx
}

// GetParam gets param.
func (a *Affine) GetParam() entity.Param {
	return a.Param
}

// GetGrad gets gradient.
func (a *Affine) GetGrad() entity.Grad {
	return a.Grad
}

func (a *Affine) SetParam(p entity.Param) {
	a.Param = p
}
