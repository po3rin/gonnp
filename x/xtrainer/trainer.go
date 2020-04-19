// Package xtrainer impliments shorhand of training for deep lerning.
package xtrainer

import (
	"fmt"
	"math/rand"
	"reflect"
	"time"

	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/params"
	"gonum.org/v1/gonum/mat"
)

// Model is neural network interface.
type Model interface {
	Forward(out chan<- float64, in ...<-chan mat.Matrix)
	Backward(out chan<- mat.Matrix)
	params.SetManager
}

// Optimizer updates prams.
type Optimizer interface {
	Update(params []params.Param, grads []params.Grad) []params.Param
}

// Train has trainer config.
type Train struct {
	Model        Model
	Optimizer    Optimizer
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
func InitTrainer(model Model, opt Optimizer, options ...OptionFunc) *Train {
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

		dx := matutil.ThinRow(x, idx)
		dt := matutil.ThinRow(teacher, idx)

		for j := 0; j < maxIters; j++ {
			bx := dx.Slice(j*batchSize, (j+1)*batchSize, 0, c)
			bt := dt.Slice(j*batchSize, (j+1)*batchSize, 0, tc)

			bxc := make(chan mat.Matrix, 1)
			btc := make(chan mat.Matrix, 1)
			lossc := make(chan float64, 1)

			go t.Model.Forward(lossc, btc, bxc)

			bxc <- bx
			btc <- bt
			loss := <-lossc

			out := make(chan mat.Matrix, 1)
			go t.Model.Backward(out)
			<-out

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

func rmDuplicate(params []params.Param, grads []params.Grad) ([]params.Param, []params.Grad) {
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
					break
				}
			}
			if findFlg {
				break
			}
		}
		if !findFlg {
			break
		}
	}
	return params, grads
}

// GetWordDist returns Words Distributed representation.
func (t *Train) GetWordDist() mat.Matrix {
	return t.Model.GetParams()[0].Weight
}
