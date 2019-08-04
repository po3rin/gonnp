// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonlp/layers"
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

	sig := layers.InitReluLayer()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sig.Forward(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}
