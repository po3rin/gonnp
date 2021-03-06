// Package matutil has utility functions of gonum matrix.
package matutil

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// SumCol calcurates sum each of columns.
func SumCol(x mat.Matrix) *mat.VecDense {
	_, c := x.Dims()

	switch m := x.(type) {
	case *mat.Dense:
		d := mat.NewVecDense(c, nil)
		for i := 0; i < c; i++ {
			d.SetVec(i, mat.Sum(m.ColView(i)))
		}
		return d

	case *mat.VecDense:
		return mat.NewVecDense(1, []float64{mat.Sum(m)})

	default:
		panic("gonnp: not yet support type")
	}
}

// SumRow calcurates sum each of rows.
func SumRow(x mat.Matrix) *mat.VecDense {
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

// Mat2VecDenseWithColMax mat to vec with column's max value.
func Mat2VecDenseWithColMax(x mat.Matrix) *mat.VecDense {
	r, _ := x.Dims()
	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to gonnp: not yet supported type matrix to dense")
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
	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to gonnp: not yet supported type matrix to dense")
	}

	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		vd := mat.NewVecDense(c, nil)
		rv := d.RowView(i)
		vd.AddVec(rv, v)
		result.SetRow(i, vd.RawVector().Data)
	}

	return result
}

// SubMatVec sub mat by vec.
func SubMatVec(x mat.Matrix, v mat.Vector) *mat.Dense {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n - v.AtVec(i)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

// MulMatVec mul mat by vec.
func MulMatVec(x mat.Matrix, v mat.Vector) *mat.Dense {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n * v.AtVec(i)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

// DivMatVec divids mat by vec.
func DivMatVec(x mat.Matrix, v mat.Vector) *mat.Dense {
	r, c := x.Dims()
	f := func(i, j int, n float64) float64 {
		return n / v.AtVec(i)
	}
	d := mat.NewDense(r, c, nil)
	d.Apply(f, x)
	return d
}

// NewRandMatrixWithSND creates random matrix according to standard normal distribution.
func NewRandMatrixWithSND(r, c int) *mat.Dense {
	a := mat.NewDense(r, c, nil)
	a.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64()*DsiredStdDev + DesiredMean
	}, a)
	return a
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
func SetColToRow(x mat.Matrix, targets []int) *mat.Dense {
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
func ExtractFromEachRows(x mat.Matrix, targets []int) *mat.VecDense {
	result := mat.NewVecDense(len(targets), nil)
	for i, v := range targets {
		result.SetVec(i, x.At(i, v))
	}
	return result
}

// ThinRowWithMat thins out rows.
func ThinRowWithMat(x mat.Matrix, thin mat.Matrix) *mat.Dense {
	_, c := x.Dims()
	r, _ := thin.Dims()
	result := mat.NewDense(r, c, nil)

	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonnp: x does not support other than *mat.Dense")
	}

	for i := 0; i < r; i++ {
		v := thin.At(i, 0)
		result.SetRow(i, d.RawRowView(int(v)))
	}
	return result
}

// OneHotVec2Index converts one-hot vector to index.
func OneHotVec2Index(x mat.Matrix) *mat.VecDense {
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
			panic("gonnp: x's element does not support other than *mat.Dense")
		}
		result.SetRow(n, d.RawRowView(i))
	}
	return result
}

// Set3D sets matrix into 3d matrix.
func Set3D(x []mat.Matrix, s mat.Matrix, t int) []mat.Matrix {
	sd, ok := s.(*mat.Dense)
	if !ok {
		panic("gonnp: s does not support other than *mat.Dense")
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
func JoinC(x mat.Matrix, y mat.Matrix) *mat.Dense {
	r, xc := x.Dims()
	_, yc := y.Dims()
	d := mat.NewDense(r, xc+yc, nil)
	d.Augment(x, y)
	return d
}

// NormoalizeVec normalizes matrix.
func NormoalizeVec(x mat.Vector) *mat.VecDense {
	r, _ := x.Dims()
	d := mat.NewVecDense(r, nil)
	d.MulElemVec(x, x)
	div := math.Sqrt(mat.Sum(d))

	d.ScaleVec(1/div, x)
	return d
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
