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

func (a *Affine) GetParamAndGrad() (entity.Param, entity.Grad) {
	return a.Param, a.Grad
}
