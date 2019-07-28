package optimizers

import "gonum.org/v1/gonum/mat"

type Optimizer interface {
	Update(params []mat.Matrix, grad []mat.Matrix) []mat.Matrix
}
