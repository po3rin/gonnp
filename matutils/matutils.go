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

// SumRow calcurates sum each of rows.
func SumRow(x mat.Matrix) mat.Vector {
	r, c := x.Dims()
	var fs []float64
	for i := 0; i < r; i++ {
		var sum float64
		for j := 0; j < c; j++ {
			sum += x.At(i, j)
		}
		fs = append(fs, sum)
	}
	return mat.NewVecDense(r, fs)
}

var (
	// DsiredStdDev used in NewRandMatrixWithSND & NewRandVecWithSND
	DsiredStdDev = 0.01
	// DesiredMean used in NewRandMatrixWithSND & NewRandVecWithSND
	DesiredMean = 0.0
)

func Mat2VecWithColMax(x mat.Matrix) mat.Vector {
	r, _ := x.Dims()
	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonlp: failed to transpose matrix to dense")
	}
	maxs := make([]float64, 0, r)
	for i := 0; i < r; i++ {
		v := d.RowView(i)
		max := mat.Max(v)
		maxs = append(maxs, max)
	}
	return mat.NewVecDense(r, maxs)
}

func SubMatVec(x mat.Matrix, v mat.Vector) mat.Matrix {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n - v.AtVec(i)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

func DivMatVec(x mat.Matrix, v mat.Vector) mat.Matrix {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n / v.AtVec(i)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

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

// At3D returns matrix. example: (6,2,7) => (6,7)
func At3D(x []mat.Matrix, i int) mat.Matrix {
	_, c := x[0].Dims()
	result := mat.NewDense(len(x), c, nil)
	for n, m := range x {
		// r, c := m.Dims()
		d, ok := m.(*mat.Dense)
		if !ok {
			panic("gonlp: failed to transpose matrix to dense")
		}
		result.SetRow(n, d.RawRowView(i))
	}
	return result
}

// Sort3DWithIDs shuffles 3 dimentional matrix col using slice of integer.
func Sort3DWithIDs(x []mat.Matrix, ids []int) []mat.Matrix {
	result := make([]mat.Matrix, len(ids))
	for i, id := range ids {
		result[i] = x[id]
	}
	return result
}

// PrintDims prints dimensions for debug.
func PrintDims(x mat.Matrix) {
	r, c := x.Dims()
	fmt.Printf("[%v, %v]\n", r, c)
}

func Print3D(x []mat.Matrix) {
	fmt.Println("=====3D Matrix======")
	for i, m := range x {
		if i != 0 {
			fmt.Println("-------------------")
		}
		PrintMat(m)
	}
	fmt.Println("====================")
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
