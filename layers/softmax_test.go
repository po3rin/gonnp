// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
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
			name:  "3*3",
			input: mat.NewDense(3, 3, []float64{1, 2, 3, 6, 5, 4, 7, 9, 8}),
			want: mat.NewDense(3, 3, []float64{
				0.09003057, 0.24472847, 0.66524096,
				0.66524096, 0.24472847, 0.09003057,
				0.09003057, 0.66524096, 0.24472847,
			}),
		},
		{
			name: "real float64",
			input: mat.NewDense(3, 7, []float64{
				1.14718039e-04, 1.41163514e-04, 1.55093818e-04, 1.77699626e-04, -6.81674514e-05, 1.30854448e-04, 1.70691753e-04,
				1.32509979e-04, 3.22600967e-06, 1.73721219e-04, -1.03128191e-04, 7.86874673e-05, 5.67694875e-05, -1.10763898e-04,
				1.52057647e-04, 2.77545642e-06, 1.59079778e-04, -8.08138788e-05, 1.85972444e-04, 1.10894458e-04, -1.20739922e-04,
			}),
			want: mat.NewDense(3, 7, []float64{
				0.14285675, 0.14286053, 0.14286252, 0.14286575, 0.14283063, 0.14285906, 0.14286475,
				0.14287136, 0.14285289, 0.14287725, 0.1428377, 0.14286367, 0.14286054, 0.14283661,
				0.14287051, 0.14284919, 0.14287152, 0.14283725, 0.14287536, 0.14286463, 0.14283154,
			}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := layers.Softmax(tt.input); !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Fatalf("unexpected data\nwant = %v\ngot = %v\n", tt.want, got)
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
		{
			name: "real float",
			input: mat.NewDense(3, 3, []float64{
				3.04847074443256e-07, 1.6644086307618e-05, 0.00090873632138,
				1.12146971388934e-07, 2.2525321346561e-06, 4.5243317361302e-05,
				6.12301716965576e-06, 0.002470201429289, 0.9965503823023,
			}),
			teacher: mat.NewDense(3, 3, []float64{0, 0, 1, 1, 0, 0, 1, 0, 0}),
			want: mat.NewDense(3, 3, []float64{
				1.01615691e-07, 5.54802877e-06, -3.33030421e-01,
				-3.33333296e-01, 7.50844045e-07, 1.50811058e-05,
				-3.33331292e-01, 8.23400476e-04, 3.32183461e-01,
			}),
		},
	}

	l := layers.InitSoftmaxWithLossLayer()
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

		{
			name: "zero-matrix",
			input: mat.NewDense(2, 2, []float64{
				0.4999581890028191, 0.5000418109971809,
				0.4999654788052471, 0.5000345211947529,
			}),
			teacher: mat.NewDense(1, 2, []float64{0, 0}),
			want:    0.6932233156766562,
		},

		{
			name: "real data in simple cbow",
			input: mat.NewDense(3, 7, []float64{
				0.0031788410528517537, 0.7568355841108841, 0.0020575610206242435, 0.23099155281373124, 0.002055489467710028, 0.0016443096882422764, 0.003236661845956284,
				0.014734365452763536, 0.5371724684861279, 0.00284645704833572, 0.4249707795682345, 0.002841432930993964, 0.002615277661266765, 0.014819218852277617,
				0.0035888475486742536, 0.004489662775245304, 0.41047245928370063, 0.0007863463542162996, 0.4096189479193935, 0.167366884090703, 0.0036768520280669446,
			}),
			teacher: mat.NewDense(3, 7, []float64{
				0, 1, 0, 0, 0, 0, 0,
				0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0,
			}),
			want: 0.6756238149145345,
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
