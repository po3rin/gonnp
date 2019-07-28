package optimizers

import "gonum.org/v1/gonum/mat"

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
func (s *SDG) Update(params []mat.Matrix, grad []mat.Matrix) []mat.Matrix {
	results := make([]mat.Matrix, 0, len(params))
	for n, p := range params {
		f := func(i, j int, v float64) float64 {
			return v - s.LR*grad[n].At(i, j)
		}
		var result mat.Dense
		result.Apply(f, p)
		results = append(results, &result)
	}
	return results
}
