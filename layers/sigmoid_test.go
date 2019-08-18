// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestSigmoidForward(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			input: mat.NewDense(2, 2, []float64{2, 2, 2, 2}),
			want: mat.NewDense(2, 2, []float64{
				0.88079707797788,
				0.88079707797788,
				0.88079707797788,
				0.88079707797788,
			}),
		},
		{
			name:  "2*2 with 0",
			input: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
			want:  mat.NewDense(2, 2, []float64{0.5, 0.5, 0.5, 0.5}),
		},
	}

	sig := layers.InitSigmoidLayer()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sig.Forward(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestSigmoidBackward(t *testing.T) {
	tests := []struct {
		name  string
		x     mat.Matrix
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			x:     mat.NewDense(2, 2, []float64{1, 1, 2, 1}),
			input: mat.NewDense(2, 2, []float64{1, 2, 4, 1}),
			want:  mat.NewDense(2, 2, []float64{0, 0, -8, 0}),
		},
	}

	sig := layers.InitSigmoidLayer()
	for _, tt := range tests {
		sig.X = tt.x
		t.Run(tt.name, func(t *testing.T) {
			if got := sig.Backward(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

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

	l := layers.InitSigmoidWithLossLayer()
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := l.Forward(tt.input, tt.teacher); got != tt.want {
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

	l := layers.InitSigmoidWithLossLayer()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l.X = tt.input
			l.Teacher = tt.teacher
			if got := l.Backward(); !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}
