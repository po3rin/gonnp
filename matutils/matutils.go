package matutils

import (
	"gonum.org/v1/gonum/mat"
)

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

func MatToFloat64(x mat.Matrix) []float64 {
	r, c := x.Dims()
	result := make([]float64, 0, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result = append(result, x.At(i, j))
		}
	}
	return result
}

func Add(a mat.Matrix, b mat.Matrix) mat.Matrix {
	var B mat.Dense
	B.Add(a, b)
	return &B
}
