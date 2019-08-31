package layers

import (
	"math"

	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/matutils"
	"gonum.org/v1/gonum/mat"
)

type rnnCache struct {
	x     mat.Matrix
	hPrev mat.Matrix
	hNext mat.Matrix
}

// RNN is RNN's layer.
type RNN struct {
	Param entity.Param
	Grad  entity.Grad
	cache rnnCache
}

// InitRNNLayer inits RNN layer.
func InitRNNLayer(wx, wh mat.Matrix, b mat.Vector) *RNN {
	return &RNN{
		Param: entity.Param{
			Weight:  wx,
			WeightH: wh,
			Bias:    b,
		},
	}
}

// Forward is RNN forward.
func (r *RNN) Forward(x, hPrev mat.Matrix) mat.Matrix {
	mr, _ := hPrev.Dims()
	_, mc := r.Param.WeightH.Dims()

	d1 := mat.NewDense(mr, mc, nil)
	d2 := mat.NewDense(mr, mc, nil)

	d1.Product(hPrev, r.Param.WeightH)
	d2.Product(x, r.Param.Weight)

	d1.Add(d1, d2)
	t := matutils.AddMatVec(d1, r.Param.Bias)

	t.Apply(func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}, t)

	r.cache = rnnCache{
		x:     x,
		hPrev: hPrev,
		hNext: t,
	}

	return t
}

// Backward is RNN backward.
func (r *RNN) Backward(dhNext mat.Matrix) (dx, dhPrev mat.Matrix) {
	mr, mc := dhNext.Dims()
	dt := mat.NewDense(mr, mc, nil)
	dt.Apply(func(i, j int, v float64) float64 {
		return v * (1 - math.Pow(r.cache.hNext.At(i, j), 2))
	}, dhNext)

	matutils.PrintMat(dt)

	// db
	db := matutils.SumCol(dt)

	// dwh
	mr, _ = r.cache.hPrev.T().Dims()
	dwh := mat.NewDense(mr, mc, nil)
	dwh.Product(r.cache.hPrev.T(), dt)

	// dhp
	mr, _ = dt.Dims()
	_, mc = r.Param.WeightH.T().Dims()
	dhp := mat.NewDense(mr, mc, nil)
	dhp.Product(dt, r.Param.WeightH.T())

	// dwx
	mr, _ = r.cache.x.T().Dims()
	_, mc = dt.Dims()
	dwx := mat.NewDense(mr, mc, nil)
	dwx.Product(r.cache.x.T(), dt)

	// dx
	mr, _ = dt.Dims()
	_, mc = r.Param.Weight.T().Dims()
	dxd := mat.NewDense(mr, mc, nil)
	dxd.Product(dt, r.Param.Weight.T())

	r.Grad.Weight = dwx
	r.Grad.WeightH = dwh
	r.Grad.Bias = db

	return dxd, dhp
}
