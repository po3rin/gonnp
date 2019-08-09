package optimizers

import (
	"github.com/po3rin/gonnp/entity"
	"gonum.org/v1/gonum/mat"
)

// SDG has setting for Stochastic Gradient Descent.
type SDG struct {
	LR float64
}

// InitSDG inits SDG setting.
// lr is learning rate.
func InitSDG(lr float64) *SDG {
	return &SDG{
		LR: lr,
	}
}

// Update updates prams using gradient.
func (s *SDG) Update(params []entity.Param, grads []entity.Grad) []entity.Param {
	for n := 0; n < len(params); n++ {
		wr, wc := grads[n].Weight.Dims()
		l := grads[n].Bias.Len()

		tmpW := mat.NewDense(wr, wc, nil)
		tmpB := mat.NewVecDense(l, nil)

		tmpW.Scale(s.LR, grads[n].Weight)
		tmpB.ScaleVec(s.LR, grads[n].Bias)

		wr, wc = params[n].Weight.Dims()
		l = params[n].Bias.Len()
		W := mat.NewDense(wr, wc, nil)
		B := mat.NewVecDense(l, nil)

		W.Sub(params[n].Weight, tmpW)
		B.SubVec(params[n].Bias, tmpB)

		params[n].Weight = W
		params[n].Bias = B
	}
	return params
}
