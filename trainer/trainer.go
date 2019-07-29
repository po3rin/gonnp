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

type Train struct {
	Model        nn.NeuralNet
	Optimizer    optimizers.Optimizer
	LossList     []interface{}
	EvalInterval int
	CurrentEpoch float64
}

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

	fmt.Println(len(options))
	for _, option := range options {
		option(t)
	}

	return t
}

func (t *Train) Fit(x mat.Matrix, teacher mat.Matrix, maxEpoch, batchSize int) {
	dataSize, c := x.Dims()
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
			bt := dt.Slice(j*batchSize, (j+1)*batchSize, 0, 1)

			// a, s := bx.Dims()
			// b, n := bt.Dims()
			// println(a)
			// println(s)
			// println(b)
			// println(n)

			loss := t.Model.Forward(bx, bt)
			t.Model.Backward(nil)
			t.Model.SetParams(t.Optimizer.Update(t.Model.GetParams(), t.Model.GetGrads()))

			totalLoss += loss
			lossCount++
		}

		t.CurrentEpoch++
	}
}
