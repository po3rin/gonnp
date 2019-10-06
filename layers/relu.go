package layers

import (
	"github.com/po3rin/gonnp/params"
	"gonum.org/v1/gonum/mat"
)

type Relu struct {
	X     mat.Matrix
	Param params.Param
	Grad  params.Grad
}

// InitReluLayer inits Relu layer.
func InitReluLayer() *Relu {
	return &Relu{}
}

func (r *Relu) Forward(x mat.Matrix) mat.Matrix {
	ro, c := x.Dims()
	d := mat.NewDense(ro, c, nil)
	relu := func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}
	d.Apply(relu, x)
	r.X = d
	return d
}

func (r *Relu) Backward(x mat.Matrix) mat.Matrix {
	ro, c := x.Dims()
	dense := mat.NewDense(ro, c, nil)
	reluBack := func(i, j int, v float64) float64 {
		if r.X.At(i, j) > 0 {
			return v
		}
		return 0
	}
	dense.Apply(reluBack, x)
	return dense
}

func (r *Relu) GetParam() params.Param {
	return r.Param
}

func (r *Relu) GetGrad() params.Grad {
	return r.Grad
}

func (r *Relu) SetParam(p params.Param) {
	r.Param = p
}
