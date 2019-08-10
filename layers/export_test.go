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
