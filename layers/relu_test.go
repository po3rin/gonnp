// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestReluForward(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			input: mat.NewDense(2, 2, []float64{-1, 2, -4, 1}),
			want:  mat.NewDense(2, 2, []float64{0, 2, 0, 1}),
		},
	}

	relu := layers.InitReluLayer()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := relu.Forward(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestReluBackward(t *testing.T) {
	tests := []struct {
		name  string
		out   mat.Matrix
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			out:   mat.NewDense(2, 2, []float64{0, 2, 0, 1}),
			input: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			want:  mat.NewDense(2, 2, []float64{0, 2, 0, 4}),
		},
	}

	relu := layers.InitReluLayer()
	for _, tt := range tests {
		relu.X = tt.out
		t.Run(tt.name, func(t *testing.T) {
			if got := relu.Backward(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}
