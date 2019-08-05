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
		{
			name:  "one-hot",
			input: mat.NewDense(2, 2, []float64{1, 0, 0, 0}),
			want:  mat.NewDense(2, 2, []float64{0.47536688641867, 0.174877704527109, 0.174877704527109, 0.174877704527109}),
		},
		{
			name:  "3*3",
			input: mat.NewDense(3, 3, []float64{6, 10, 14, 5, 8, 11, 9, 15, 21}),
			want: mat.NewDense(3, 3, []float64{
				3.04847074443256e-07, 1.6644086307618e-05, 0.00090873632138,
				1.12146971388934e-07, 2.2525321346561e-06, 4.5243317361302e-05,
				6.12301716965576e-06, 0.002470201429289, 0.9965503823023,
			}),
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

// TODO: cofirm correct want.
func TestSoftmaxWithLossForward(t *testing.T) {
	tests := []struct {
		name    string
		input   mat.Matrix
		teacher mat.Matrix
		want    float64
	}{
		{
			name:    "1*3 with one-hot",
			input:   mat.NewDense(1, 3, []float64{0.3, 2.9, 4}),
			teacher: mat.NewDense(1, 3, []float64{0, 1, 0}),
			want:    1.405714056968575,
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

func TestSoftmaxWithLossBackward(t *testing.T) {
	tests := []struct {
		name    string
		input   mat.Matrix
		teacher mat.Matrix
		want    mat.Matrix
	}{
		{
			name:    "1*3",
			input:   mat.NewDense(1, 3, []float64{1, 0, 0}),
			teacher: mat.NewDense(1, 3, []float64{1, 0, 0}),
			want:    mat.NewDense(1, 3, []float64{0, 0, 0}),
		},
		{
			name:    "1*3",
			input:   mat.NewDense(1, 3, []float64{0.01, 0.99, 0}),
			teacher: mat.NewDense(1, 3, []float64{0, 1, 0}),
			want:    mat.NewDense(1, 3, []float64{0.01, -0.01, 0}),
		},
	}

	l := layers.InitSoftmaxWithLossLayer()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l.X = tt.input
			l.Teacher = tt.teacher
			if got := l.Backward(); !mat.EqualApprox(got, tt.want, 1e-14) {
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
			input:   mat.NewDense(1, 10, []float64{0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0}),
			teacher: mat.NewDense(1, 10, []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
			want:    0.51082545709933802,
		},

		{
			name:    "one-hot-2",
			input:   mat.NewDense(1, 10, []float64{0.1, 0.05, 0.1, 0, 0.05, 0.1, 0, 0.6, 0, 0}),
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
