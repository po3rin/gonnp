// Package trainer impliments shorhand of training for deep lerning.
package trainer

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/po3rin/gonlp/matutils"
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

		// shuffle x
		tx := matutils.Shuffle3D(x)

		// shuffle t
		rand.Seed(time.Now().UnixNano())
		idx := rand.Perm(dataSize)
		tt := matutils.ThinCol(teacher, idx)
		dt, ok := tt.(*mat.Dense)
		if !ok {
			panic("gonlp: failed to transpose matrix to dense")
		}

		for j := 0; j < maxIters; j++ {
			bx := tx[j*batchSize : (j+1)*batchSize]
			bt := dt.Slice(j*batchSize, (j+1)*batchSize, 0, tc)

			loss := t.Model.Forward(bt, bx...)
			t.Model.Backward()
			t.Model.UpdateParams(
				t.Optimizer.Update(
					t.Model.GetParams(), t.Model.GetGrads(),
				),
			)

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
