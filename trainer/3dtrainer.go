// Package trainer impliments shorhand of training for deep lerning.
package trainer

import (
	"fmt"
	"math/rand"

	"github.com/po3rin/gonnp/matutil"
	"gonum.org/v1/gonum/mat"
)

// Fit3D traims from data using 3 dimentional matrix.
func (t *Train) Fit3D(x []mat.Matrix, teacher mat.Matrix, maxEpoch, batchSize int) {
	dataSize := len(x)
	_, tc := teacher.Dims()

	maxIters := int(dataSize / batchSize)
	var totalLoss float64
	var lossCount int

	for i := 0; i < maxEpoch; i++ {
		idx := rand.Perm(dataSize)

		// shuffle x
		tx := matutil.Sort3DWithIDs(x, idx)

		// shuffle t
		dt := matutil.ThinRow(teacher, idx)

		for j := 0; j < maxIters; j++ {
			bx := tx[j*batchSize : (j+1)*batchSize]
			bt := dt.Slice(j*batchSize, (j+1)*batchSize, 0, tc)

			loss := t.Model.Forward(bt, bx...)
			t.Model.Backward()

			params := t.Model.GetParams()
			grads := t.Model.GetGrads()

			params, grads = rmDuplicate(params, grads)
			params = t.Optimizer.Update(params, grads)

			t.Model.UpdateParams(params)

			totalLoss += loss
			lossCount++

			if j%t.EvalInterval == 0 {
				avgLoss := totalLoss / float64(lossCount)
				fmt.Printf("| epoch %v |  iter %v / %v | loss %.4f\n", t.CurrentEpoch+1, j+1, maxIters, avgLoss)
				t.LossList = append(t.LossList, float64(avgLoss))
				totalLoss, lossCount = 0, 0
			}
		}
		t.CurrentEpoch++
	}
}

// GetWordDist returns Words Distributed representation.
func (t *Train) GetWordDist() mat.Matrix {
	return t.Model.GetParams()[0].Weight
}
