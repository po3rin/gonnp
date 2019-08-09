// +build !e2e

package matutils_test

import (
	"reflect"
	"testing"

	"github.com/po3rin/gonnp/matutils"
	"gonum.org/v1/gonum/mat"
)

func TestSumCol(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			input: mat.NewDense(2, 2, []float64{2, 2, 2, 2}),
			want:  mat.NewVecDense(2, []float64{4, 4}),
		},
		{
			name:  "2*2 with 0",
			input: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
			want:  mat.NewVecDense(2, []float64{0, 0}),
		},
		{
			name:  "3*2",
			input: mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}),
			want:  mat.NewVecDense(2, []float64{9, 12}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.SumCol(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestSumRow(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			input: mat.NewDense(2, 2, []float64{2, 2, 2, 2}),
			want:  mat.NewVecDense(2, []float64{4, 4}),
		},
		{
			name:  "2*2 with 0",
			input: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
			want:  mat.NewVecDense(2, []float64{0, 0}),
		},
		{
			name:  "3*2",
			input: mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}),
			want:  mat.NewVecDense(3, []float64{3, 7, 11}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.SumRow(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestMat2VecWithColMax(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			input: mat.NewDense(2, 2, []float64{2, 4, 4, 2}),
			want:  mat.NewVecDense(2, []float64{4, 4}),
		},
		{
			name:  "2*2",
			input: mat.NewDense(3, 3, []float64{1, 4, 2, 2, 7, 3, 8, 2, 1}),
			want:  mat.NewVecDense(3, []float64{4, 7, 8}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.Mat2VecWithColMax(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestSubMatVec(t *testing.T) {
	tests := []struct {
		name string
		mat  mat.Matrix
		vec  mat.Vector
		want mat.Matrix
	}{
		{
			name: "2*2",
			mat:  mat.NewDense(2, 2, []float64{2, 4, 4, 2}),
			vec:  mat.NewVecDense(2, []float64{3, 1}),
			want: mat.NewDense(2, 2, []float64{-1, 1, 3, 1}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.SubMatVec(tt.mat, tt.vec); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestThinRow(t *testing.T) {
	tests := []struct {
		name   string
		input  mat.Matrix
		target []int
		want   mat.Matrix
	}{
		{
			name:   "4*2",
			input:  mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8}),
			target: []int{0, 3},
			want:   mat.NewDense(2, 2, []float64{1, 2, 7, 8}),
		},
		{
			name:   "4*2 with no sort",
			input:  mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8}),
			target: []int{3, 0},
			want:   mat.NewDense(2, 2, []float64{7, 8, 1, 2}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.ThinRow(tt.input, tt.target); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d\n", tt.want, got)
			}
		})
	}
}

func TestOneHotVec2Index(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "4*2",
			input: mat.NewDense(4, 2, []float64{1, 0, 1, 0, 0, 1, 0, 1}),
			want:  mat.NewVecDense(4, []float64{0, 0, 1, 1}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.OneHotVec2Index(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d\n", tt.want, got)
			}
		})
	}
}

func TestNewRandMatrixWithSND(t *testing.T) {
	tests := []struct {
		name string
		r, c int
	}{
		{
			name: "3*3",
			r:    3,
			c:    3,
		},
		{
			name: "2*3",
			r:    2,
			c:    3,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			got := matutils.NewRandMatrixWithSND(tt.r, tt.c)
			if r, c := got.Dims(); r != tt.r || c != tt.c {
				t.Fatalf("want = [%v, %v], got = [%v, %v]\n", tt.r, tt.c, r, c)
			}
		})
	}
}

func TestNewRandVecWithSND(t *testing.T) {
	tests := []struct {
		name string
		r    int
	}{
		{
			name: "3",
			r:    3,
		},
		{
			name: "2",
			r:    2,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			got := matutils.NewRandVecWithSND(tt.r, nil)
			if r, _ := got.Dims(); r != tt.r {
				t.Fatalf("want = %v, got = %v\n", tt.r, r)
			}
		})
	}
}

func TestAt3D(t *testing.T) {
	tests := []struct {
		name string
		at   int
		x    []mat.Matrix
		want mat.Matrix
	}{
		{
			name: "simple",
			at:   0,
			x: []mat.Matrix{
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 1, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 1, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 0, 1, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 0, 0, 1, 0,
				}),
			},
			want: mat.NewDense(6, 7, []float64{
				0, 1, 0, 0, 0, 0, 0,
				0, 0, 1, 0, 0, 0, 0,
				0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0,
				0, 1, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 1, 0,
			}),
		},
		{
			name: "3 dimention",
			at:   1,
			x: []mat.Matrix{
				mat.NewDense(2, 7, []float64{
					1, 0, 0, 0, 0, 0, 0,
					0, 0, 1, 0, 0, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 1, 0, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 0, 1, 0, 0, 0, 0,
					0, 0, 0, 0, 1, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 0, 0, 1, 0, 0, 0,
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 0, 0, 0, 1, 0, 0,
					0, 0, 0, 0, 0, 1, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 1,
				}),
			},
			want: mat.NewDense(6, 7, []float64{
				0, 0, 1, 0, 0, 0, 0,
				0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0,
				0, 1, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 1, 0,
				0, 0, 0, 0, 0, 0, 1,
			}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			got := matutils.At3D(tt.x, tt.at)
			if !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Errorf("x:\nwant = %d\ngot = %d", tt.want, got)
			}
		})
	}
}

func TestSort3DWithIDs(t *testing.T) {
	tests := []struct {
		name    string
		ids     []int
		x       []mat.Matrix
		notwant []mat.Matrix
	}{
		{
			name: "6*1*7 dimention",
			ids:  []int{3, 1, 4, 2, 0, 5},
			x: []mat.Matrix{
				mat.NewDense(1, 7, []float64{
					1, 0, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 1, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 1, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 0, 1, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
			},
			notwant: []mat.Matrix{
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 1, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 0, 1, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 1, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					1, 0, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			got := matutils.Sort3DWithIDs(tt.x, tt.ids)

			if !reflect.DeepEqual(got, tt.notwant) {
				t.Error("not sort")
			}
		})
	}
}

func TestPrint(t *testing.T) {
	d := mat.NewDense(1, 1, nil)
	matutils.PrintMat(d)
	matutils.PrintDims(d)
	matutils.Print3D([]mat.Matrix{d})
}
