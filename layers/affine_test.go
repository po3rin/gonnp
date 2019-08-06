// +build !e2e

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
				weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			},
			want: want{
				x:      mat.NewDense(2, 2, []float64{5, 11, 11, 25}),
				bias:   mat.NewVecDense(2, []float64{4, 6}),
				weight: mat.NewDense(2, 2, []float64{10, 14, 14, 20}),
			},
		},
		{
			name: "real float v1",
			input: input{
				out: mat.NewDense(3, 3, []float64{
					1.01615691e-07, 5.54802877e-06, -3.33030421e-01,
					-3.33333296e-01, 7.50844045e-07, 1.50811058e-05,
					-3.33331292e-01, 8.23400476e-04, 3.32183461e-01,
				}),
				x:      mat.NewDense(3, 2, []float64{2, 4, 2, 3, 3, 6}),
				weight: mat.NewDense(2, 3, []float64{1, 1, 1, 1, 2, 3}),
			},
			want: want{
				x: mat.NewDense(3, 2, []float64{
					-3.33024772e-01, -9.99080066e-01,
					-3.33317464e-01, -3.33286551e-01,
					-3.24431084e-04, 6.64865891e-01,
				}),
				bias: mat.NewVecDense(3, []float64{-0.66666449, 0.0008297, -0.00083188}),
				weight: mat.NewDense(2, 3, []float64{
					-1.66666027e+00, 2.48279917e-03, 3.30519702e-01,
					-2.99998724e+00, 4.96484751e-03, 6.61024323e-01,
				}),
			},
		},

		{
			name: "real float v2",
			input: input{
				out: mat.NewDense(3, 2, []float64{
					-3.33024772e-01, -9.99080066e-01,
					-3.33317464e-01, -3.33286551e-01,
					-3.24431084e-04, 6.64865891e-01,
				}),
				x:      mat.NewDense(3, 3, []float64{1, 0, 1, 1, 1, 0, 1, 1, 1}),
				weight: mat.NewDense(3, 2, []float64{1, 1, 1, 2, 1, 3}),
			},
			want: want{
				x: mat.NewDense(3, 3, []float64{
					-1.33210484, -2.3311849, -3.33026497,
					-0.66660401, -0.99989057, -1.33317712,
					0.66454146, 1.32940735, 1.99427324,
				}),
				bias: mat.NewVecDense(2, []float64{-0.66666667, -0.66750073}),
				weight: mat.NewDense(3, 2, []float64{
					-0.66666667, -0.66750073,
					-0.3336419, 0.33157934,
					-0.3333492, -0.33421418,
				}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			aff := layers.InitAffineLayer(tt.input.weight, tt.input.bias)
			aff.X = tt.input.x

			got := aff.Backward(tt.input.out)
			if !mat.EqualApprox(got, tt.want.x, 1e-7) {
				t.Errorf("x:\nwant = %d\ngot = %d", tt.want.x, got)
			}
			if !mat.EqualApprox(aff.GetGrad().Weight, tt.want.weight, 1e-7) {
				t.Errorf("grad.weight:\nwant = %d\ngot = %d", tt.want.weight, aff.GetGrad().Weight)
			}
			if !mat.EqualApprox(aff.GetGrad().Bias, tt.want.bias, 1e-7) {
				t.Errorf("grad.bias\nwant = %d\ngot = %d", tt.want.bias, aff.GetGrad().Bias)
			}
		})
	}
}
