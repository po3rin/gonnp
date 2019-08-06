// Package matutils has utility functions of gonum matrix.
package matutils

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// SumCol calcurates sum each of columns.
func SumCol(x mat.Matrix) mat.Vector {
	r, c := x.Dims()
	A := mat.NewVecDense(c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum := A.AtVec(j) + x.At(i, j)
			A.SetVec(j, sum)
		}
	}
	return A
}

var (
	// DsiredStdDev used in NewRandMatrixWithSND & NewRandVecWithSND
	DsiredStdDev = 0.1
	// DesiredMean used in NewRandMatrixWithSND & NewRandVecWithSND
	DesiredMean = 0.0
)

// NewRandMatrixWithSND creates random matrix according to standard normal distribution.
func NewRandMatrixWithSND(r, c int) mat.Matrix {
	a := make([]float64, 0, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			rand.Seed(time.Now().UnixNano())
			a = append(a, rand.NormFloat64()*DsiredStdDev+DesiredMean)
		}
	}
	return mat.NewDense(r, c, a)
}

// NewRandVecWithSND creates random vector according to standard normal distribution.
func NewRandVecWithSND(r int, _ []float64) *mat.VecDense {
	a := make([]float64, 0, r)
	for i := 0; i < r; i++ {
		rand.Seed(time.Now().UnixNano())
		a = append(a, rand.NormFloat64()*DsiredStdDev+DesiredMean)
	}
	return mat.NewVecDense(r, a)
}

// ThinCol thins out rows.
// TODO: rm type assertion.
func ThinCol(x mat.Matrix, targets []int) mat.Matrix {
	// 	sort.Ints(targets)
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

// OneHotVec2Index converts one-hot vector to index.
func OneHotVec2Index(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	a := make([]float64, 0, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if x.At(i, j) == 1 {
				a = append(a, float64(j))
			}
		}
	}
	return mat.NewVecDense(r, a)
}

// PrintDims prints dimensions for debug.
func PrintDims(x mat.Matrix) {
	r, c := x.Dims()
	fmt.Printf("[%v, %v]\n", r, c)
}

// CheckNaNOrInf checks whether matrix has nan or inf element.
func CheckNaNOrInf(x mat.Matrix, label interface{}) {
	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonlp: failed to transpose matrix to dense")
	}
	if math.IsNaN(mat.Sum(d)) {
		fmt.Printf("=====%v====\n", label)
		panic("nan!!")
	}
	if math.IsInf(mat.Sum(d), 0) {
		fmt.Printf("=====%v====\n", label)
		panic("inf!!")
	}
}

// PrintMat prints matrix formatted.
func PrintMat(x mat.Matrix) {
	fa := mat.Formatted(x, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
