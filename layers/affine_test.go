package layers_test

import (
	"testing"

	"github.com/po3rin/gonlp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestAffineForward(t *testing.T) {
	type input struct {
		data   mat.Matrix
		weight mat.Matrix
		bias   mat.Matrix
	}
	tests := []struct {
		name  string
		input input
		want  mat.Matrix
	}{
		{
			name: "normal",
			input: input{
				data:   mat.NewDense(2, 3, []float64{1, 1, 1, 1, 1, 1}),
				weight: mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}),
				bias:   mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
			},
			want: mat.NewDense(2, 2, []float64{10, 12, 9, 13}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			aff := layers.InitAffineLayer(tt.input.weight, tt.input.bias)
			if got := aff.Forward(tt.input.data); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}
