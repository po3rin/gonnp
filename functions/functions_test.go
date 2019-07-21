package functions_test

import (
	"testing"

	"github.com/po3rin/gonlp/functions"
	"gonum.org/v1/gonum/mat"
)

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
			if got := functions.CrossEntropyErr(tt.input, tt.teacher); got != tt.want {
				t.Fatalf("want = %v, got = %v", tt.want, got)
			}
		})
	}
}
