// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
	"github.com/po3rin/gonnp/word"
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

func TestEmbeddingDotBackward(t *testing.T) {
	tests := []struct {
		name         string
		weight       mat.Matrix
		dout         mat.Matrix
		idx          mat.Matrix
		cacheH       mat.Matrix
		cacheTargetW mat.Matrix
		want         mat.Matrix
	}{
		{
			name:         "simple",
			weight:       mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8}),
			dout:         mat.NewVecDense(4, []float64{1, 2, 3, 4}),
			idx:          mat.NewVecDense(4, []float64{1, 0, 3, 0}),
			cacheH:       mat.NewDense(4, 2, []float64{1, 1, 2, 2, 3, 3, 4, 4}),
			cacheTargetW: mat.NewDense(4, 2, []float64{3, 4, 1, 2, 7, 8, 1, 2}),
			want:         mat.NewDense(4, 2, []float64{3, 4, 2, 4, 21, 24, 4, 8}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := layers.InitEmbeddingDotLayer(tt.weight)

			e.SetCacheH(tt.cacheH)
			e.SetCacheTargetW(tt.cacheTargetW)

			e.Embed.IDx = tt.idx

			got := e.Backward(tt.dout)

			if !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Fatalf("unexpected data\nwant = %v\ngot = %v\n", tt.want, got)
			}

			if !mat.EqualApprox(got, tt.want, 1e-7) {
				t.Fatalf("unexpected data\nwant = %v\ngot = %v\n", tt.want, got)
			}
		})
	}
}

func TestGetNegativeSample(t *testing.T) {
	tests := []struct {
		power      float64
		sampleSize int
		corpus     word.Corpus
		target     mat.Vector
		want       mat.Matrix
	}{
		{
			power:      0.75,
			sampleSize: 2,
			corpus: word.Corpus{
				0, 1, 2, 3, 4, 1, 2, 3,
			},
			target: mat.NewVecDense(3, []float64{1, 3, 0}),
			want:   mat.NewDense(3, 2, []float64{2, 3, 1, 2, 2, 3}),
		},
	}

	randGenerator := func(max float64) float64 {
		return 0.3
	}
	layers.UseCustomRandGenerator(randGenerator)

	for _, tt := range tests {
		u := layers.InitUnigraSampler(tt.corpus, tt.power, tt.sampleSize)
		got := u.GetNegativeSample(tt.target)

		if !mat.EqualApprox(got, tt.want, 1e-7) {
			t.Errorf("x:\nwant = %d\ngot = %d", tt.want, got)
		}
	}
}
