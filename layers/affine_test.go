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
		bias   mat.Vector
	}
	tests := []struct {
		name  string
		input input
		want  mat.Matrix
	}{
		{
			name: "normal",
			input: input{
				data:   mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
				weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				bias:   mat.NewVecDense(2, []float64{1, 1}),
			},
			want: mat.NewDense(2, 2, []float64{5, 7, 5, 7}),
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

func TestAffineBackword(t *testing.T) {
	type input struct {
		out    mat.Matrix
		x      mat.Matrix
		bias   mat.Vector
		weight mat.Matrix
	}
	type want struct {
		x      mat.Matrix
		bias   mat.Vector
		weight mat.Matrix
	}
	tests := []struct {
		name  string
		input input
		want  want
	}{
		{
			name: "normal",
			input: input{
				out:    mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				x:      mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				bias:   mat.NewVecDense(2, []float64{1, 2}),
				weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			},
			want: want{
				x:      mat.NewDense(2, 2, []float64{5, 11, 11, 25}),
				bias:   mat.NewVecDense(2, []float64{4, 6}),
				weight: mat.NewDense(2, 2, []float64{10, 14, 14, 20}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			aff := layers.InitAffineLayer(tt.input.weight, tt.input.bias)
			aff.X = tt.input.out

			got := aff.Backward(tt.input.x)
			if !mat.EqualApprox(got, tt.want.x, 1e-14) {
				t.Errorf("want = %d, got = %d", tt.want.x, got)
			}
			if !mat.EqualApprox(aff.GetGrad().Weight, tt.want.weight, 1e-14) {
				t.Errorf("want = %d, got = %d", tt.want.weight, aff.GetGrad().Weight)
			}
			if !mat.EqualApprox(aff.GetGrad().Bias, tt.want.bias, 1e-14) {
				t.Errorf("want = %d, got = %d", tt.want.bias, aff.GetGrad().Bias)
			}
		})
	}
}
