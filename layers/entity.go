package layers

import "gonum.org/v1/gonum/mat"

type Grad struct {
	Weight mat.Matrix
	Bias   mat.Matrix
}

type Param struct {
	Weight mat.Matrix
	Bias   mat.Matrix
}
