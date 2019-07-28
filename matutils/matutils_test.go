package matutils_test

import (
	"testing"

	"github.com/po3rin/gonlp/matutils"
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
			want:  mat.NewDense(1, 2, []float64{4, 4}),
		},
		{
			name:  "2*2 with 0",
			input: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
			want:  mat.NewDense(1, 2, []float64{0, 0}),
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

func TestThinCol(t *testing.T) {
	tests := []struct {
		name   string
		input  mat.Matrix
		target []int
		want   mat.Matrix
	}{
		{
			name:   "2*2",
			input:  mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8}),
			target: []int{0, 3},
			want:   mat.NewDense(2, 2, []float64{1, 2, 7, 8}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.ThinCol(tt.input, tt.target); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}
