package functions

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// CrossEntropyErr measures the performance of a classification model whose output is a probability value between 0 and 1.
func CrossEntropyErr(data mat.Matrix, teacher mat.Matrix) float64 {
	// TODO: if teacher data is one-hot, ignore 0 in teacher data.
	batchSize, _ := data.Dims()
	var ce mat.Dense
	ce.Apply(crossEnrtopy, data)

	var mul mat.Dense
	mul.MulElem(teacher, &ce)

	sum := mat.Sum(&mul)
	return -sum / float64(batchSize)
}

func crossEnrtopy(i, j int, v float64) float64 {
	delta := 0.0000001
	return math.Log(v + delta)
}
