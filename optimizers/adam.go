package optimizers

import (
	"math"

	"github.com/po3rin/gonlp/entity"
	"gonum.org/v1/gonum/mat"
)

// Adsm has setting for Adam optimizer.
type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	M     []mat.Matrix
	V     []mat.Matrix
	Iter  float64
}

// InitAdam inits Adam optimizer.
func InitAdam(lr, beta1, beta2 float64) *Adam {
	return &Adam{
		LR:    lr,
		Beta1: beta1,
		Beta2: beta2,
		M:     []mat.Matrix{},
		V:     []mat.Matrix{},
	}
}

// Update updates params using Adam argolism. supports weight only.
func (a *Adam) Update(params []entity.Param, grads []entity.Grad) []entity.Param {
	pow := func(i, j int, v float64) float64 {
		return math.Pow(v, 2)
	}
	sqrtWithMin := func(i, j int, v float64) float64 {
		return math.Sqrt(v) + 1e-7
	}

	if len(a.M) == 0 {
		for _, p := range params {
			r, c := p.Weight.Dims()
			a.M = append(a.M, mat.NewDense(r, c, nil))
			a.V = append(a.V, mat.NewDense(r, c, nil))
		}
	}

	a.Iter++
	lrT := a.LR * math.Sqrt(1.0-math.Pow(a.Beta2, a.Iter)) / (1.0 - math.Pow(a.Beta1, a.Iter))

	result := make([]entity.Param, len(params))
	for i := range params {

		// m
		r, c := grads[i].Weight.Dims()
		mb := mat.NewDense(r, c, nil)
		mb.Sub(grads[i].Weight, a.M[i])
		mb.Scale(1-a.Beta1, mb)
		mb.Add(a.M[i], mb)
		m := mb

		// v
		mb.Apply(pow, grads[i].Weight)
		mb.Sub(mb, a.V[i])
		mb.Scale(1-a.Beta2, mb)
		mb.Add(a.V[i], mb)
		v := mb

		// set m & v
		a.M[i] = m
		a.V[i] = v

		// set new params
		mb.Apply(sqrtWithMin, a.V[i])
		mb.DivElem(a.M[i], mb)
		mb.Scale(lrT, mb)
		mb.Sub(params[i].Weight, mb)
		result[i].Weight = mb
	}
	return result
}
