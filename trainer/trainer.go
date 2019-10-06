// Package trainer impliments shorhand of training for deep lerning.
package trainer

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/po3rin/gonnp/params"
	"github.com/po3rin/gonnp/matutils"
	"github.com/po3rin/gonnp/optimizers"
	"gonum.org/v1/gonum/mat"
)

// NeuralNet is neural network interface.
type NeuralNet interface {
	Forward(teacher mat.Matrix, x ...mat.Matrix) float64
	Backward() mat.Matrix
	params.SetManager
}

// Train has trainer config.
type Train struct {
	Model        NeuralNet
	Optimizer    optimizers.Optimizer
	LossList     []float64
	EvalInterval int
	CurrentEpoch float64
}

// OptionFunc for set option for trainer
type OptionFunc func(t *Train)

// EvalInterval sets EvalInterval option.
func EvalInterval(i int) func(*Train) {
	return func(t *Train) {
		t.EvalInterval = i
	}
}

// InitTrainer inits Trainer.
func InitTrainer(model NeuralNet, opt optimizers.Optimizer, options ...OptionFunc) *Train {
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

// Fit traims from data.
func (t *Train) Fit(x mat.Matrix, teacher mat.Matrix, maxEpoch, batchSize int) {
	dataSize, c := x.Dims()
	_, tc := teacher.Dims()

	maxIters := int(dataSize / batchSize)
	var totalLoss float64
	var lossCount int

	for i := 0; i < maxEpoch; i++ {
		// shuffle
		rand.Seed(time.Now().UnixNano())
		idx := rand.Perm(dataSize)

		dx := matutils.ThinRow(x, idx)
		dt := matutils.ThinRow(teacher, idx)

		for j := 0; j < maxIters; j++ {
			bx := dx.Slice(j*batchSize, (j+1)*batchSize, 0, c)
			bt := dt.Slice(j*batchSize, (j+1)*batchSize, 0, tc)

			loss := t.Model.Forward(bt, bx)
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
				fmt.Printf("| epoch %v |  iter %v / %v | loss %.4f\n", t.CurrentEpoch, j, maxIters, avgLoss)
				t.LossList = append(t.LossList, float64(avgLoss))
				totalLoss, lossCount = 0, 0
			}
		}
		t.CurrentEpoch++
	}
}
