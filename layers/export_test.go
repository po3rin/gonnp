// +build !e2e

package layers

import "gonum.org/v1/gonum/mat"

var CrossEntropyErr = crossEntropyErr
var Softmax = softmax

func (e *EmbeddingDot) ExportCacheH() mat.Matrix {
	return e.cache.h
}

func (e *EmbeddingDot) ExportCacheTargetW() mat.Matrix {
	return e.cache.targetW
}

func (e *EmbeddingDot) SetCacheH(h mat.Matrix) {
	e.cache.h = h
}

func (e *EmbeddingDot) SetCacheTargetW(targetW mat.Matrix) {
	e.cache.targetW = targetW
}

func UseCustomRandGenerator(f func(max float64) float64) (resetFunc func()) {
	var tmp func(max float64) float64
	tmp, randGenerator = randGenerator, f
	return func() {
		randGenerator = tmp
	}
}

func (r *RNN) SetRNNCacheX(x mat.Matrix) {
	r.cache.x = x
}

func (r *RNN) SetRNNCacheHPrev(h mat.Matrix) {
	r.cache.hPrev = h
}

func (r *RNN) SetRNNCacheNextPrev(h mat.Matrix) {
	r.cache.hNext = h
}
