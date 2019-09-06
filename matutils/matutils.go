// Package matutils has utility functions of gonum matrix.
package matutils

import (
	"fmt"
	"math"
	"math/rand"

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

// Mat2VecWithColMax mat to vec with column's max value.
func Mat2VecWithColMax(x mat.Matrix) mat.Vector {
	r, _ := x.Dims()
	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to transpose matrix to dense")
	}
	maxs := make([]float64, 0, r)
	for i := 0; i < r; i++ {
		v := d.RowView(i)
		max := mat.Max(v)
		maxs = append(maxs, max)
	}
	return mat.NewVecDense(r, maxs)
}

// AddMatVec add mat + vec.
func AddMatVec(x mat.Matrix, v mat.Vector) *mat.Dense {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n + v.AtVec(j)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

// SubMatVec sub mat by vec.
func SubMatVec(x mat.Matrix, v mat.Vector) mat.Matrix {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n - v.AtVec(i)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

// MulMatVec mul mat by vec.
func MulMatVec(x mat.Matrix, v mat.Vector) mat.Matrix {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n * v.AtVec(i)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

// DivMatVec divids mat by vec.
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
			a = append(a, rand.NormFloat64()*DsiredStdDev+DesiredMean)
		}
	}
	return mat.NewDense(r, c, a)
}

// NewRandVecWithSND creates random vector according to standard normal distribution.
func NewRandVecWithSND(r int, _ []float64) *mat.VecDense {
	a := make([]float64, 0, r)
	for i := 0; i < r; i++ {
		a = append(a, rand.NormFloat64()*DsiredStdDev+DesiredMean)
	}
	return mat.NewVecDense(r, a)
}

// ThinRow thins out rows.
func ThinRow(x mat.Matrix, targets []int) *mat.Dense {
	_, c := x.Dims()
	result := mat.NewDense(len(targets), c, nil)

	d, ok := x.(*mat.Dense)
	if !ok {
		d = mat.DenseCopyOf(x)
	}

	for i, v := range targets {
		result.SetRow(i, d.RawRowView(v))
	}
	return result
}

// SetColToRow set row to row using targets.
func SetColToRow(x mat.Matrix, targets []int) mat.Matrix {
	r, _ := x.Dims()
	result := mat.NewDense(r, len(targets), nil)
	d := mat.DenseCopyOf(x)

	for i, v := range targets {
		s := make([]float64, 0, len(targets))
		for j := 0; j < r; j++ {
			s = append(s, d.At(j, v))
		}
		result.SetRow(i, s)
	}
	return result
}

// ExtractFromEachRows extracts from row from targets.
func ExtractFromEachRows(x mat.Matrix, targets []int) mat.Matrix {
	result := mat.NewDense(1, len(targets), nil)
	for i, v := range targets {
		result.Set(0, i, x.At(i, v))
	}
	return result
}

// ThinRowWithMat thins out rows.
func ThinRowWithMat(x mat.Matrix, thin mat.Matrix) mat.Matrix {
	_, c := x.Dims()
	r, _ := thin.Dims()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		d, ok := x.(*mat.Dense)
		if !ok {
			panic("gonnp: failed to transpose matrix to dense")
		}
		v := thin.At(i, 0)
		result.SetRow(i, d.RawRowView(int(v)))
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

// New3D creates 3D matrix.
func New3D(N, T, D int) []mat.Matrix {
	hs := make([]mat.Matrix, N)

	for i := range hs {
		hs[i] = mat.NewDense(T, D, nil)
	}
	return hs
}

// At3D returns matrix. example: (6,2,7) => (6,7)
func At3D(x []mat.Matrix, i int) *mat.Dense {
	_, c := x[0].Dims()
	result := mat.NewDense(len(x), c, nil)
	for n, m := range x {
		d, ok := m.(*mat.Dense)
		if !ok {
			panic("gonnp: failed to transpose matrix to dense")
		}
		result.SetRow(n, d.RawRowView(i))
	}
	return result
}

// Set3D sets matrix into 3d matrix.
func Set3D(x []mat.Matrix, s mat.Matrix, t int) []mat.Matrix {
	sd, ok := s.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to transpose matrix to dense")
	}
	for i, m := range x {
		md := mat.DenseCopyOf(m)
		v := sd.RawRowView(i)
		md.SetRow(t, v)
		x[i] = md
	}
	return x
}

// Sort3DWithIDs shuffles 3 dimentional matrix col using slice of integer.
func Sort3DWithIDs(x []mat.Matrix, ids []int) []mat.Matrix {
	result := make([]mat.Matrix, len(ids))
	for i, id := range ids {
		result[i] = x[id]
	}
	return result
}

// Reshape3DTo2D reshapes (3,4,5) -> (12,5)
func Reshape3DTo2D(x []mat.Matrix) *mat.Dense {
	T, D := x[0].Dims()
	fs := make([]float64, 0, len(x)*T*D)
	for _, m := range x {
		var md mat.Dense
		md.CloneFrom(m)
		fs = append(fs, md.RawMatrix().Data...)
	}
	return mat.NewDense(len(x)*T, D, fs)
}

// Reshape2DTo3D rechapes if r=2 (12,3) -> (6,2,3)
func Reshape2DTo3D(x mat.Matrix, s int) []mat.Matrix {
	r, D := x.Dims()
	N := int(r / s)

	var md mat.Dense
	md.CloneFrom(x)

	rs := make([]mat.Matrix, N)
	for i := 0; i < N; i++ {
		rs[i] = md.Slice(i*s, i*s+s, 0, D)
	}
	return rs
}

// JoinC join matrix.
func JoinC(x mat.Matrix, y mat.Matrix) mat.Matrix {
	xr, xc := x.Dims()
	yr, yc := y.Dims()

	if xr != yr {
		panic("gonnp: row length mismatch")
	}

	yd := mat.DenseCopyOf(y)
	xd := mat.DenseCopyOf(x)
	expanded := xd.Grow(0, yc)
	ed := mat.DenseCopyOf(expanded)

	for i := 0; i < yc; i++ {
		v := yd.ColView(i)
		vd := mat.VecDenseCopyOf(v)
		ed.SetCol(xc+i, vd.RawVector().Data)
	}

	return ed
}

// NormoalizeVec normalizes matrix.
func NormoalizeVec(x *mat.VecDense) *mat.VecDense {
	r, _ := x.Dims()
	d := mat.NewVecDense(r, nil)
	d.MulElemVec(x, x)
	div := math.Sqrt(mat.Sum(d))

	x.ScaleVec(1/div, x)
	return x
}

// PrintDims prints dimensions for debug.
func PrintDims(x mat.Matrix) {
	r, c := x.Dims()
	fmt.Printf("[%v, %v]\n", r, c)
}

// Print3D print 3D matrix (slice of matrix)
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

// PrintMat prints matrix formatted.
func PrintMat(x mat.Matrix) {
	fa := mat.Formatted(x, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
