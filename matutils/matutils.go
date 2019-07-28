package matutils

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// SumCol calcurates sum each of columns.
func SumCol(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	A := mat.NewDense(1, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum := A.At(0, j) + x.At(i, j)
			A.Set(0, j, sum)
		}
	}
	return A
}

// NewRandMatrixWithSND creates random matrix according to standard normal distribution.
func NewRandMatrixWithSND(r, c int) mat.Matrix {
	a := make([]float64, 0, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			a = append(a, rand.NormFloat64())
		}
	}
	return mat.NewDense(r, c, a)
}

// ThinCol thins out rows.
// TODO: rm type assertion.
func ThinCol(x mat.Matrix, targets []int) mat.Matrix {
	_, c := x.Dims()
	result := mat.NewDense(len(targets), c, nil)

	for i, v := range targets {
		d, ok := x.(*mat.Dense)
		if !ok {
			panic("gonlp: failed to transpose matrix to dense")
		}
		result.SetRow(i, d.RawRowView(v))
	}
	return result
}
