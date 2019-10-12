package layers

import (
	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/params"
	"gonum.org/v1/gonum/mat"
)

// Affine layers perform the linear transformation
type Affine struct {
	X     mat.Matrix
	Param params.Param
	Grad  params.Grad
}

// InitAffineLayer inits affine layer.
func InitAffineLayer(weight mat.Matrix, bias mat.Vector) *Affine {
	return &Affine{
		Param: params.Param{
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
	db := matutil.SumCol(x)

	a.Grad.Weight = &dw
	a.Grad.Bias = db

	return &dx
}

// GetParam gets param.
func (a *Affine) GetParam() params.Param {
	return a.Param
}

// GetGrad gets gradient.
func (a *Affine) GetGrad() params.Grad {
	return a.Grad
}

func (a *Affine) SetParam(p params.Param) {
	a.Param = p
}

// TimeAffine layers perform the linear transformation.
type TimeAffine struct {
	X     []mat.Matrix
	Param params.Param
	Grad  params.Grad
}

// InitTimeAffineLayer inits affine layer.
func InitTimeAffineLayer(weight mat.Matrix, bias mat.Vector) *TimeAffine {
	return &TimeAffine{
		Param: params.Param{
			Weight: weight,
			Bias:   bias,
		},
	}
}

// Forward for time affine layer.
func (a *TimeAffine) Forward(x []mat.Matrix) []mat.Matrix {
	a.X = x
	T, _ := x[0].Dims()

	rx := matutil.Reshape3DTo2D(x)

	r, _ := rx.Dims()
	_, c := a.Param.Weight.Dims()
	b := mat.NewDense(r, c, nil)
	b.Product(rx, a.Param.Weight)

	b.Apply(func(i, j int, v float64) float64 {
		return v + a.Param.Bias.AtVec(j)
	}, b)

	return matutil.Reshape2DTo3D(b, T)
}

// Backward for time affine layer.
func (a *TimeAffine) Backward(dout []mat.Matrix) []mat.Matrix {
	T, _ := a.X[0].Dims()
	m := matutil.Reshape3DTo2D(dout)
	rx := matutil.Reshape3DTo2D(a.X)

	// dw
	r, _ := rx.T().Dims()
	_, c := m.Dims()
	dw := mat.NewDense(r, c, nil)
	dw.Product(rx.T(), m)

	// db
	db := matutil.SumCol(m)

	// dx
	r, _ = m.Dims()
	_, c = a.Param.Weight.T().Dims()
	d := mat.NewDense(r, c, nil)
	d.Product(m, a.Param.Weight.T())
	dx := matutil.Reshape2DTo3D(d, T)

	a.Grad.Weight = dw
	a.Grad.Bias = db

	return dx
}
