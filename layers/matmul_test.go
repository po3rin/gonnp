// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestMatMulForward(t *testing.T) {
	type input struct {
		data   mat.Matrix
		weight mat.Matrix
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
			},
			want: mat.NewDense(2, 2, []float64{4, 6, 4, 6}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			matmul := layers.InitMatMulLayer(tt.input.weight)
			if got := matmul.Forward(tt.input.data); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestMatMulBackword(t *testing.T) {
	type input struct {
		out    mat.Matrix
		x      mat.Matrix
		weight mat.Matrix
	}
	type want struct {
		x      mat.Matrix
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
				weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			},
			want: want{
				x:      mat.NewDense(2, 2, []float64{5, 11, 11, 25}),
				weight: mat.NewDense(2, 2, []float64{10, 14, 14, 20}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			matmul := layers.InitMatMulLayer(tt.input.weight)
			matmul.X = tt.input.x

			got := matmul.Backward(tt.input.out)
			if !mat.EqualApprox(got, tt.want.x, 1e-7) {
				t.Errorf("x:\nwant = %d\ngot = %d", tt.want.x, got)
			}
			if !mat.EqualApprox(matmul.GetGrad().Weight, tt.want.weight, 1e-7) {
				t.Errorf("grad.weight:\nwant = %d\ngot = %d", tt.want.weight, matmul.GetGrad().Weight)
			}
		})
	}
}
