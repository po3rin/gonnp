package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestEmbeddingForward(t *testing.T) {
	tests := []struct {
		name   string
		data   mat.Matrix
		weight mat.Matrix
		want   mat.Matrix
	}{
		{
			name: "simple",
			data: mat.NewVecDense(4, []float64{1, 0, 3, 0}),
			weight: mat.NewDense(7, 3, []float64{
				0, 1, 2,
				3, 4, 5,
				6, 7, 8,
				9, 10, 11,
				12, 13, 14,
				15, 16, 17,
				18, 19, 20,
			}),
			want: mat.NewDense(4, 3, []float64{
				3, 4, 5,
				0, 1, 2,
				9, 10, 11,
				0, 1, 2,
			}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := layers.InitEmbeddingLayer(tt.weight)
			if got := e.Forward(tt.data); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestEmbeddingBackward(t *testing.T) {
	tests := []struct {
		name     string
		weight   mat.Matrix
		idh      mat.Matrix
		idx      mat.Matrix
		wantGrad mat.Matrix
	}{
		{
			name: "simple",
			weight: mat.NewDense(8, 3, []float64{
				0, 1, 2,
				3, 4, 5,
				6, 7, 8,
				9, 10, 11,
				0, 1, 2,
				3, 4, 5,
				6, 7, 8,
				9, 10, 11,
			}),

			idh: mat.NewDense(4, 3, []float64{
				0, 1, 2,
				3, 4, 5,
				6, 7, 8,
				9, 10, 11,
			}),
			idx: mat.NewVecDense(4, []float64{1, 0, 3, 0}),
			wantGrad: mat.NewDense(8, 3, []float64{
				12, 14, 16,
				0, 1, 2,
				0, 0, 0,
				6, 7, 8,
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
			}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := layers.InitEmbeddingLayer(tt.weight)
			e.IDx = tt.idx
			_ = e.Backward(tt.idh)
			got := e.Grad.Weight
			if !mat.EqualApprox(got, tt.wantGrad, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.wantGrad, got)
			}
		})
	}
}

func TestTimeEmbeddingForward(t *testing.T) {
	tests := []struct {
		name   string
		data   mat.Matrix
		weight mat.Matrix
		want   []mat.Matrix
	}{
		{
			name: "simple",
			data: mat.NewDense(3, 2, []float64{
				2, 3,
				4, 1,
				5, 0,
			}),
			weight: mat.NewDense(7, 2, []float64{
				-0.07756537, -0.42653856,
				-0.77619475, 0.36609185,
				0.5545087, 0.8681872,
				-0.17239346, 0.70797604,
				-0.3200279, -0.6484259,
				0.69929963, -0.42481396,
				-0.00797722, -0.01127309,
			}),
			want: []mat.Matrix{
				mat.NewDense(2, 2, []float64{
					0.5545087, 0.8681872,
					-0.17239346, 0.70797604,
				}),
				mat.NewDense(2, 2, []float64{
					-0.3200279, -0.6484259,
					-0.77619475, 0.36609185,
				}),
				mat.NewDense(2, 2, []float64{
					0.69929963, -0.42481396,
					-0.07756537, -0.42653856,
				}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := layers.InitTimeEmbeddingLayer(tt.weight)
			got := e.Forward(tt.data)
			for i, m := range got {
				if !mat.EqualApprox(m, tt.want[i], 1e-7) {
					t.Errorf("dx:\nwant = %d\ngot = %d", m, tt.want[i])
				}
			}
		})
	}
}
