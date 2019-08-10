// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestEmbeddingDotForward(t *testing.T) {
	tests := []struct {
		name             string
		weight           mat.Matrix
		h                mat.Matrix
		idx              mat.Matrix
		want             mat.Matrix
		wantCacheH       mat.Matrix
		wantCacheTargetW mat.Matrix
	}{
		{
			name:             "simple",
			weight:           mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8}),
			h:                mat.NewDense(4, 2, []float64{1, 1, 2, 2, 3, 3, 4, 4}),
			idx:              mat.NewVecDense(4, []float64{1, 0, 3, 0}),
			want:             mat.NewVecDense(4, []float64{7, 6, 45, 12}),
			wantCacheH:       mat.NewDense(4, 2, []float64{1, 1, 2, 2, 3, 3, 4, 4}),
			wantCacheTargetW: mat.NewDense(4, 2, []float64{3, 4, 1, 2, 7, 8, 1, 2}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := layers.InitEmbeddingDotLayer(tt.weight)
			got := e.Forward(tt.h, tt.idx)

			if !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Fatalf("unexpected data\nwant = %v\ngot = %v\n", tt.want, got)
			}

			cacheH := e.ExportCacheH()
			cacheTargetW := e.ExportCacheTargetW()

			if !mat.EqualApprox(cacheH, tt.wantCacheH, 1e-7) {
				t.Fatalf("unexpected data\nwant = %v\ngot = %v\n", tt.wantCacheH, cacheH)
			}
			if !mat.EqualApprox(cacheTargetW, tt.wantCacheTargetW, 1e-7) {
				t.Fatalf("unexpected data\nwant = %v\ngot = %v\n", tt.wantCacheTargetW, cacheTargetW)
			}
		})
	}
}
