// +build !e2e

package xlayers_test

import (
	"testing"

	"github.com/po3rin/gonnp/x/xlayers"
	"gonum.org/v1/gonum/mat"
)

func TestSigmoidWithLossForward(t *testing.T) {
	tests := []struct {
		name    string
		input   mat.Matrix
		teacher mat.Matrix
		want    float64
	}{
		{
			name:    "1*5 with one-hot",
			input:   mat.NewDense(5, 1, []float64{0, 1, 4, 0, 1}),
			teacher: mat.NewDense(5, 1, []float64{0, 0, 1, 0, 0}),
			want:    0.80619328371728,
			// want:    6.830962146228082,
			// want: 0.8036288736236525,
		},
	}

	l := xlayers.InitSigmoidWithLossLayer()
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			in := make(chan mat.Matrix, 1)
			teacher := make(chan mat.Matrix, 1)
			out := make(chan float64, 1)

			go l.Forward(out, in, teacher)

			in <- tt.input
			teacher <- tt.teacher
			got := <-out

			if got != tt.want {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}

func TestSigmoidWithLossBackward(t *testing.T) {
	tests := []struct {
		name    string
		input   mat.Matrix
		teacher mat.Matrix
		want    mat.Matrix
	}{
		{
			name: "1*5",
			input: mat.NewDense(5, 1, []float64{
				0.5,
				0.7310585786300049,
				0.9820137900379085,
				0.5,
				0.7310585786300049,
			}),
			teacher: mat.NewDense(5, 1, []float64{0, 0, 1, 0, 0}),
			want:    mat.NewDense(5, 1, []float64{0.1, 0.14621172, -0.00359724, 0.1, 0.14621172}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := xlayers.InitSigmoidWithLossLayer()
			l.X = tt.input
			l.Teacher = tt.teacher

			out := make(chan mat.Matrix, 1)

			go l.Backward(out)

			got := <-out

			if !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}
