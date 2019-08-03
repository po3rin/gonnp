// Package trainer impliments shorhand of training for deep lerning.
package trainer

import (
	"fmt"
	"math/rand"

	"github.com/po3rin/gonlp/matutils"
	"github.com/po3rin/gonlp/nn"
	"github.com/po3rin/gonlp/optimizers"
	"gonum.org/v1/gonum/mat"
)

// Train has trainer config.
type Train struct {
	Model        nn.NeuralNet
	Optimizer    optimizers.Optimizer
	LossList     []float64
	EvalInterval int
	CurrentEpoch float64
}

// OptionFunc for set option for trainer
type OptionFunc func(t *Train)

func EvalInterval(i int) func(*Train) {
	return func(t *Train) {
		t.EvalInterval = i
	}
}

func InitTrainer(model nn.NeuralNet, opt optimizers.Optimizer, options ...OptionFunc) *Train {
	t := &Train{
		Model:     model,
		Optimizer: opt,
		// set default value.
		EvalInterval: 20,
	}

	for _, option := range options {
		option(t)
	}

	return t
}

func (t *Train) Fit(x mat.Matrix, teacher mat.Matrix, maxEpoch, batchSize int) {
	dataSize, c := x.Dims()
	_, tc := teacher.Dims()

	maxIters := int(dataSize / batchSize)
	var totalLoss float64
	var lossCount int

	for i := 0; i < maxEpoch; i++ {

		// shuffle
		idx := rand.Perm(dataSize)
		tx := matutils.ThinCol(x, idx)
		tt := matutils.ThinCol(teacher, idx)

		dx, ok := tx.(*mat.Dense)
		if !ok {
			panic("gonlp: failed to transpose matrix to dense")
		}
		dt, ok := tt.(*mat.Dense)
		if !ok {
			panic("gonlp: failed to transpose matrix to dense")
		}

		for j := 0; j < maxIters; j++ {
			bx := dx.Slice(j*batchSize, (j+1)*batchSize, 0, c)
			bt := dt.Slice(j*batchSize, (j+1)*batchSize, 0, tc)

			loss := t.Model.Forward(bx, bt)
			t.Model.Backward()
			t.Model.SetParams(t.Optimizer.Update(t.Model.GetParams(), t.Model.GetGrads()))
			t.Model.UpdateParams()

			totalLoss += loss
			lossCount++

			if j%t.EvalInterval == 0 {
				avgLoss := totalLoss / float64(lossCount)
				fmt.Printf("| epoch %v |  iter %v / %v | loss %.4f\n", t.CurrentEpoch, j, maxIters, avgLoss)
				t.LossList = append(t.LossList, float64(avgLoss))
				totalLoss, lossCount = 0, 0
			}
			t.CurrentEpoch++
		}
	}
}
