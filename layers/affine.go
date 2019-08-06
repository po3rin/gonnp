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
	// matutils.CheckNaNOrInf(x, "x in for")
	a.X = x

	// matutils.CheckNaNOrInf(a.Param.Weight, "w in for")

	var b mat.Dense
	// fmt.Println("=================")
	// fmt.Println(mat.Max(a.Param.Weight))
	// fmt.Println(mat.Max(x))

	b.Product(x, a.Param.Weight)

	// matutils.CheckNaNOrInf(&b, "b in for")

	var c mat.Dense
	add := func(i, j int, v float64) float64 {
		return v + a.Param.Bias.AtVec(j)
	}
	c.Apply(add, &b)

	// matutils.CheckNaNOrInf(&c, "c in for")
	return &c
}

// Backward for affine layer.
func (a *Affine) Backward(x mat.Matrix) mat.Matrix {
	// fmt.Println("=======before======")
	// matutils.PrintMat(x)
	// matutils.PrintMat(a.Param.Weight)
	// matutils.PrintMat(a.X)
	// matutils.PrintMat(a.Param.Bias)
	// fmt.Println("======after=======")
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
