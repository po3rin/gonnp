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

type SamplerMock struct {
}

func newSamplerMock() *SamplerMock {
	return &SamplerMock{}
}

func (s *SamplerMock) GetNegativeSample(target mat.Vector) mat.Matrix {
	return mat.NewDense(2, 3, []float64{
		1, 3, 0,
		0, 2, 3,
	})
}

func TestNegativeSamplingLossForward(t *testing.T) {
	tests := []struct {
		corpus     word.Corpus
		power      float64
		sampleSize int
		weight     mat.Matrix
		h          mat.Matrix
		target     mat.Matrix
		want       float64
	}{
		{
			corpus: word.Corpus{
				0, 1, 2, 1, 3,
			},
			power:      0.75,
			sampleSize: 3,
			want:       2.7726502900273906,
			weight: mat.NewDense(4, 5, []float64{
				-1.4318770e-03, 1.2904286e-02, -2.9491405e-03, 4.9344229e-04, -5.8435639e-03,
				-5.3556040e-03, -7.0864302e-03, -2.7031412e-03, 9.1236187e-03, 1.1242336e-03,
				2.2635944e-03, -3.4989098e-06, 6.3422308e-03, -8.9476391e-04, -1.7053705e-02,
				8.8509894e-04, -3.1941470e-03, 9.5496379e-04, -2.9861021e-03, -4.7442266e-03,
			}),
			h: mat.NewDense(2, 5, []float64{
				-0.00780955, -0.0127758, 0.00211832, 0.00473524, -0.00230565,
				0.00119231, -0.00038715, -0.009422, 0.01920843, -0.01840022,
			}),
			target: mat.NewDense(2, 1, []float64{
				2, 1,
			}),
		},
	}

	sampler := newSamplerMock()

	for _, tt := range tests {
		u := layers.InitNegativeSamplingLoss(tt.weight, tt.corpus, sampler, tt.power, tt.sampleSize)
		got := u.Forward(tt.h, tt.target)
		if got != tt.want {
			t.Errorf("x:\nwant = %v\ngot = %v", tt.want, got)
		}
	}
}
