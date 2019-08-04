// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonlp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "1*3",
			input: mat.NewDense(1, 3, []float64{0.3, 2.9, 4}),
			want:  mat.NewDense(1, 3, []float64{0.018211273295547, 0.24519181293507, 0.73659691376937}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := layers.Softmax(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}

func TestSoftmaxWithLossForward(t *testing.T) {
	tests := []struct {
		name    string
		input   mat.Matrix
		teacher mat.Matrix
		want    float64
	}{
		{
			name:    "1*3 with same",
			input:   mat.NewDense(1, 3, []float64{0.3, 2.9, 4}),
			teacher: mat.NewDense(1, 3, []float64{0.3, 2.9, 4}),
			want:    6.5011407735378555,
		},
	}

	l := layers.InitSoftmaxWithLossLayer()
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := l.Forward(tt.input, tt.teacher); got != tt.want {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}

func TestCrossEntropyErr(t *testing.T) {
	tests := []struct {
		name    string
		input   mat.Matrix
		teacher mat.Matrix
		want    float64
	}{
		{
			name:    "one-hot-1",
			input:   mat.NewDense(1, 10, []float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0, 0.1, 0, 0}),
			teacher: mat.NewDense(1, 10, []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
			want:    0.51082545709933802,
		},

		{
			name:    "one-hot-2",
			input:   mat.NewDense(1, 10, []float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0, 0.6, 0, 0}),
			teacher: mat.NewDense(1, 10, []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
			want:    2.3025840929945458,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := layers.CrossEntropyErr(tt.input, tt.teacher); got != tt.want {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}
