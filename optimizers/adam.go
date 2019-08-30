package optimizers

import (
	"math"
	"sync"

	"github.com/po3rin/gonnp/entity"
	"gonum.org/v1/gonum/mat"
)

// Adam has setting for Adam optimizer.
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

func powElem(i, j int, v float64) float64 {
	return math.Pow(v, 2)
}

func sqrtWithMin(i, j int, v float64) float64 {
	return math.Sqrt(v) + 1e-7
}

// Update updates params using Adam argolism. supports weight only.
func (a *Adam) Update(params []entity.Param, grads []entity.Grad) []entity.Param {
	if len(a.M) == 0 {
		for _, p := range params {
			r, c := p.Weight.Dims()
			a.M = append(a.M, mat.NewDense(r, c, nil))
			a.V = append(a.V, mat.NewDense(r, c, nil))
		}
	}

	a.Iter++
	lrT := a.LR * math.Sqrt(1.0-math.Pow(a.Beta2, a.Iter)) / (1.0 - math.Pow(a.Beta1, a.Iter))

	var wg sync.WaitGroup
	result := make([]entity.Param, len(params))

	for i := range params {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			// m
			r, c := grads[i].Weight.Dims()
			mb := mat.NewDense(r, c, nil)
			mb.Sub(grads[i].Weight, a.M[i])
			mb.Scale(1-a.Beta1, mb)
			mb.Add(a.M[i], mb)

			// v
			r, c = grads[i].Weight.Dims()
			vb := mat.NewDense(r, c, nil)
			vb.Apply(powElem, grads[i].Weight)
			vb.Sub(vb, a.V[i])
			vb.Scale(1-a.Beta2, vb)
			vb.Add(a.V[i], vb)

			// set m & v
			a.M[i] = mb
			a.V[i] = vb

			// set new params
			d := mat.NewDense(r, c, nil)
			d.Apply(sqrtWithMin, a.V[i])
			d.DivElem(a.M[i], d)
			d.Scale(lrT, d)
			d.Sub(params[i].Weight, d)
			result[i].Weight = d
		}(i)
	}

	wg.Wait()

	return result
}
