package layers

import (
	"math"

	"github.com/po3rin/gonnp/params"
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
	Param params.Param
	Grad  params.Grad
	cache rnnCache
}

// InitRNNLayer inits RNN layer.
func InitRNNLayer(wx, wh mat.Matrix, b mat.Vector) *RNN {
	return &RNN{
		Param: params.Param{
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

// TimeRNN is Time TNN layer
type TimeRNN struct {
	Param    params.Param
	Grad     params.Grad
	Layers   []*RNN
	H        mat.Matrix
	Dh       mat.Matrix
	Stateful bool
}

// InitTimeRNNLayer inits time RNN layer.
func InitTimeRNNLayer(wx, wh mat.Matrix, b mat.Vector, stateful bool) *TimeRNN {
	return &TimeRNN{
		Param: params.Param{
			Weight:  wx,
			WeightH: wh,
			Bias:    b,
		},
		Stateful: stateful,
	}
}

// Forward TimeRNN forward layer.
func (t *TimeRNN) Forward(xs []mat.Matrix) []mat.Matrix {
	T, _ := xs[0].Dims()
	N := len(xs)
	_, H := t.Param.Weight.Dims()
	hs := make([]mat.Matrix, N)

	for i := range hs {
		hs[i] = mat.NewDense(T, H, nil)
	}

	if !t.Stateful || t.H == nil {
		t.H = mat.NewDense(len(xs), H, nil)
	}

	for i := 0; i < T; i++ {
		l := InitRNNLayer(t.Param.Weight, t.Param.WeightH, t.Param.Bias)
		t.H = l.Forward(matutils.At3D(xs, i), t.H)
		matutils.Set3D(hs, t.H, i)
		t.Layers = append(t.Layers, l)
	}

	return hs
}

// Backward TimeRNN forward layer.
func (t *TimeRNN) Backward(dhs []mat.Matrix) []mat.Matrix {
	N := len(dhs)
	T, _ := dhs[0].Dims()
	D, _ := t.Param.Weight.Dims()

	dxs := make([]mat.Matrix, N)
	for i := range dxs {
		dxs[i] = mat.NewDense(T, D, nil)
	}

	var (
		dh, dx mat.Matrix
		wGrad  *mat.Dense
		whGrad *mat.Dense
		bGrad  *mat.VecDense
	)

	for i := len(t.Layers) - 1; i >= 0; i-- {
		l := t.Layers[i]
		a := matutils.At3D(dhs, i)
		if dh != nil {
			a.Add(a, dh)
		}
		dx, dh = l.Backward(a)
		matutils.Set3D(dxs, dx, i)

		wGrad.Add(wGrad, l.Grad.Weight)
		whGrad.Add(whGrad, l.Grad.WeightH)
		bGrad.AddVec(bGrad, l.Grad.Bias)
	}

	t.Grad.Weight = wGrad
	t.Grad.WeightH = whGrad
	t.Grad.Bias = bGrad

	t.Dh = dh
	return dxs
}

// SetState sets state h.
func (t *TimeRNN) SetState(h mat.Matrix) {
	t.H = h
}

// ResetState resets state h.
func (t *TimeRNN) ResetState() {
	t.H = nil
}
