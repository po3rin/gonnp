package layers_test

import (
	"testing"

	"github.com/po3rin/gonlp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestSoftmaxForward(t *testing.T) {
	tests := []struct {
		name  string
		input mat.Matrix
		want  mat.Matrix
	}{
		{
			name:  "2*2",
			input: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			want: mat.NewDense(2, 2, []float64{
				0.03205860328008,
				0.08714431874203,
				0.23688281808991,
				0.64391425988797,
			}),
		},
		{
			name:  "2*2 with 0",
			input: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
			want:  mat.NewDense(2, 2, []float64{0.25, 0.25, 0.25, 0.25}),
		},
	}

	sm := layers.InitSoftmaxLayer()
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := sm.Forward(tt.input); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}
