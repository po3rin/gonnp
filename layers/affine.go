package layers

import (
	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

// Affine layers perform the linear transformation
type Affine struct {
	X      mat.Matrix
	Weight mat.Matrix
	Bias   mat.Matrix
	Grads  mat.Matrix
}

// InitAffineLayer inits affine layer.
func InitAffineLayer(weight mat.Matrix, bias mat.Matrix) *Affine {
	return &Affine{
		Weight: weight,
		Bias:   bias,
	}
}

// Forward for affine layer.
func (a *Affine) Forward(x mat.Matrix) mat.Matrix {
	var b mat.Dense
	b.Product(x, a.Weight)

	var c mat.Dense
	c.Add(&b, a.Bias)

	a.X = &c
	return &c
}

// Backward for affine layer.
func (a *Affine) Backward(x, dout mat.Matrix) mat.Matrix {
	var dw mat.Dense
	dw.Product(a.X.T(), dout)

	db := matutils.SumCol(dout)
	_, c := db.Dims()
	grads := mat.NewDense(2, c, nil)
	grads.SetRow(0, matutils.MatToFloat64(&dw))
	grads.SetRow(1, matutils.MatToFloat64(db))

	var dx mat.Dense
	dx.Product(dout, a.Weight.T())
	return &dx
}
