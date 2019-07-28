package trainer

import (
	"math/rand"

	"github.com/po3rin/gonlp/matutils"
	"github.com/po3rin/gonlp/nn"
	"github.com/po3rin/gonlp/optimizers"
	"gonum.org/v1/gonum/mat"
)

type Train struct {
	Model         nn.NeralNet
	Optimizer     optimizers.Optimizer
	LossList      []interface{}
	EvalInterval  int
	Current_epoch float64
}

type OptionFunc func(t *Train)

func SetEvalInterval(i int) func(*Train) {
	return func(t *Train) {
		t.EvalInterval = i
	}
}

func InitTrainer(model nn.NeralNet, opt optimizers.Optimizer, options ...OptionFunc) *Train {
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
	maxIters := int(dataSize / batchSize)
	// var totalLoss int
	// var loss_count int

	for i := 0; i < maxEpoch; i++ {

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
			bt := dt.Slice(j*batchSize, (j+1)*batchSize, 0, c)

			loss := t.Model.Forward(bx, bt)
			t.Model.Backward(nil)
		}
	}
}
