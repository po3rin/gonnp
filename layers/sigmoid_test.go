// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonlp/layers"
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
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := sig.Forward(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}
