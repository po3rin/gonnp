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

func TestAddMatVec(t *testing.T) {
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
			want: mat.NewDense(2, 2, []float64{5, 5, 7, 3}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.AddMatVec(tt.mat, tt.vec); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %v, got = %v", tt.want, got)
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
				t.Fatalf("want = %v, got = %v\n", tt.want, got)
			}
		})
	}
}

func TestSetColToRow(t *testing.T) {
	tests := []struct {
		name   string
		input  mat.Matrix
		target []int
		want   mat.Matrix
	}{
		{
			name: "4*2",
			input: mat.NewDense(2, 4, []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
			}),
			target: []int{0, 2},
			want:   mat.NewDense(2, 2, []float64{1, 5, 3, 7}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.SetColToRow(tt.input, tt.target); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d\n", tt.want, got)
			}
		})
	}
}

func TestExtractFromEachRows(t *testing.T) {
	tests := []struct {
		name   string
		input  mat.Matrix
		target []int
		want   mat.Matrix
	}{
		{
			name: "4*2",
			input: mat.NewDense(3, 3, []float64{
				0, 1, 2,
				4, 5, 6,
				8, 9, 10,
			}),
			target: []int{0, 2, 2},
			want:   mat.NewDense(1, 3, []float64{0, 6, 10}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.ExtractFromEachRows(tt.input, tt.target); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d\n", tt.want, got)
			}
		})
	}
}

func TestThinRowWithMat(t *testing.T) {
	tests := []struct {
		name   string
		input1 mat.Matrix
		input2 mat.Vector
		want   mat.Matrix
	}{
		{
			name:   "normal",
			input1: mat.NewDense(4, 2, []float64{2, 0, 4, 0, 0, 6, 0, 8}),
			input2: mat.NewVecDense(2, []float64{0, 2}),
			want:   mat.NewDense(2, 2, []float64{2, 0, 0, 6}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.ThinRowWithMat(tt.input1, tt.input2); !mat.EqualApprox(got, tt.want, 1e-14) {
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

func TestMulMatVec(t *testing.T) {
	tests := []struct {
		name     string
		inputMat mat.Matrix
		inputVec mat.Vector
		want     mat.Matrix
	}{
		{
			name:     "normal",
			inputMat: mat.NewDense(2, 4, []float64{2, 0, 4, 0, 0, 6, 0, 8}),
			inputVec: mat.NewVecDense(2, []float64{2, 3}),
			want:     mat.NewDense(2, 4, []float64{4, 0, 8, 0, 0, 18, 0, 24}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.MulMatVec(tt.inputMat, tt.inputVec); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d\n", tt.want, got)
			}
		})
	}
}

func TestDivMatVec(t *testing.T) {
	tests := []struct {
		name     string
		inputMat mat.Matrix
		inputVec mat.Vector
		want     mat.Matrix
	}{
		{
			name:     "normal",
			inputMat: mat.NewDense(2, 4, []float64{2, 0, 4, 0, 0, 6, 0, 9}),
			inputVec: mat.NewVecDense(2, []float64{2, 3}),
			want:     mat.NewDense(2, 4, []float64{1, 0, 2, 0, 0, 2, 0, 3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.DivMatVec(tt.inputMat, tt.inputVec); !mat.EqualApprox(got, tt.want, 1e-14) {
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

func TestJoinC(t *testing.T) {
	tests := []struct {
		name string
		x    mat.Matrix
		y    mat.Matrix
		want mat.Matrix
	}{
		{
			name: "4*2",
			x:    mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			y:    mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			want: mat.NewDense(2, 4, []float64{1, 2, 1, 2, 3, 4, 3, 4}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.JoinC(tt.x, tt.y); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d\n", tt.want, got)
			}
		})
	}
}

func TestNormoalizeVec(t *testing.T) {
	tests := []struct {
		name string
		x    *mat.VecDense
		want *mat.VecDense
	}{
		{
			name: "4*2",
			x:    mat.NewVecDense(5, []float64{1, 2, 3, 1, 4}),
			want: mat.NewVecDense(5, []float64{0.1796053, 0.3592106, 0.53881591, 0.1796053, 0.71842121}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.NormoalizeVec(tt.x); !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Fatalf("want = %v, got = %v\n", tt.want, got)
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
