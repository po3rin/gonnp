package optimizers

import (
	"fmt"

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
		a1, a2 := params[n].Weight.Dims()
		a3, a4 := grads[n].Weight.Dims()

		fmt.Println("====================")
		fmt.Printf("[%v, %v]\n", a1, a2)
		fmt.Printf("[%v, %v]\n", a3, a4)

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
