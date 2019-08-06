// +build !e2e

package nn

import "gonum.org/v1/gonum/mat"

func UseCustomWightGenerator(f func(i, j int) mat.Matrix) (resetFunc func()) {
	var tmp func(i, j int) mat.Matrix
	tmp, weightGenerator = weightGenerator, f
	return func() {
		weightGenerator = tmp
	}
}

func UseCustomBiasGenerator(f func(i int, v []float64) *mat.VecDense) (resetFunc func()) {
	var tmp func(i int, v []float64) *mat.VecDense
	tmp, biasGenerator = biasGenerator, f
	return func() {
		biasGenerator = tmp
	}
}
