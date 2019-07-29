package optimizers

import (
	"github.com/po3rin/gonlp/entity"
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
		wr, wc := params[n].Weight.Dims()
		br, bc := params[n].Bias.Dims()

		W := mat.NewDense(wr, wc, nil)
		B := mat.NewDense(br, bc, nil)

		W.Scale(s.LR, grads[n].Weight)
		B.Scale(s.LR, grads[n].Bias)

		W.Sub(params[n].Weight, W)
		B.Sub(params[n].Bias, B)

		params[n].Weight = W
		params[n].Bias = B
	}
	return params
}
