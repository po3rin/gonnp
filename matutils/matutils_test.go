package matutils_test

import (
	"reflect"
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

func TestMatToFloat64(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  []float64
	}{
		{
			name:  "2*2",
			input: mat.NewDense(2, 2, []float64{2, 2, 2, 2}),
			want:  []float64{2, 2, 2, 2},
		},
		{
			name:  "2*2 with 0",
			input: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
			want:  []float64{0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := matutils.MatToFloat64(tt.input); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}
