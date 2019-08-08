// Package trainer impliments shorhand of training for deep lerning.
package trainer

import (
	"fmt"
	"reflect"

	"github.com/po3rin/gonlp/entity"
	"gonum.org/v1/gonum/mat"
)

// TODO: impliments
func rmDuplicate(params []entity.Param, grads []entity.Grad) ([]entity.Param, []entity.Grad) {
L:
	for {
		var findFlg bool
		L := len(params)
		for i := 0; i < L; i++ {
			for j := i + 1; j < L; j++ {
				if reflect.DeepEqual(params[i].Weight, params[j].Weight) {
					r, c := grads[i].Weight.Dims()
					d := mat.NewDense(r, c, nil)
					d.Add(grads[i].Weight, grads[j].Weight)
					grads[i].Weight = d
					findFlg = true
					params = append(params[:j], params[j+1:]...)
					grads = append(grads[:j], grads[j+1:]...)
				}
				if findFlg {
					break L
				}
			}
			if findFlg {
				break L
			}
		}
		if !findFlg {
			break L
		}
	}
	return params, grads
}

// Fit3D traims from data using 3 dimentional matrix.
func (t *Train) Fit3D(x []mat.Matrix, teacher mat.Matrix, maxEpoch, batchSize int) {
	dataSize := len(x)
	_, tc := teacher.Dims()

	maxIters := int(dataSize / batchSize)
	var totalLoss float64
	var lossCount int

	for i := 0; i < maxEpoch; i++ {

		// shuffle x
		// tx := matutils.Shuffle3D(x)
		tx := x

		// shuffle t
		// rand.Seed(time.Now().UnixNano())
		// idx := rand.Perm(dataSize)
		// tt := matutils.ThinCol(teacher, idx)
		tt := teacher
		dt, ok := tt.(*mat.Dense)
		if !ok {
			panic("gonlp: failed to transpose matrix to dense")
		}

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
