package layers

import (
	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

// Affine layers perform the linear transformation
type Affine struct {
	X     mat.Matrix
	Param entity.Param
	Grad  entity.Grad
}

// InitAffineLayer inits affine layer.
func InitAffineLayer(weight mat.Matrix, bias mat.Matrix) *Affine {
	return &Affine{
		Param: entity.Param{
			Weight: weight,
			Bias:   bias,
		},
	}
}

// Forward for affine layer.
func (a *Affine) Forward(x mat.Matrix) mat.Matrix {
	a1, a2 := x.Dims()
	a3, a4 := a.Param.Weight.Dims()
	println(a1)
	println(a2)
	println(a3)
	println(a4)

	var b mat.Dense
	b.Product(x, a.Param.Weight)

	var c mat.Dense
	c.Add(&b, a.Param.Bias)

	a.X = &c
	return &c
}

// Backward for affine layer.
func (a *Affine) Backward(x mat.Matrix) mat.Matrix {
	var dw mat.Dense
	dw.Product(a.X.T(), x)

	db := matutils.SumCol(x)

	a.Grad.Weight = &dw
	a.Grad.Bias = db

	var dx mat.Dense
	dx.Product(x, a.Param.Weight.T())
	return &dx
}

func (a *Affine) GetParam() entity.Param {
	return a.Param
}

func (a *Affine) GetGrad() entity.Grad {
	return a.Grad
}
