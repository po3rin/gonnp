// +build e2e

package e2e_test

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/models"
	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/trainer"
	"github.com/po3rin/gonnp/word"
)

func TestSimpleCBOW(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	windowSize := 1
	hiddenSize := 5
	batchSize := 3
	maxEpoch := 1000

	text := "You say goodbye and I say hello."
	corpus, w2id, id2w := word.PreProcess(text)

	vocabSize := len(w2id)
	contexts, target := word.CreateContextsAndTarget(corpus, windowSize)

	te := word.ConvertOneHot(target, vocabSize)
	co := word.ConvertOneHot(contexts, vocabSize)

	model := models.InitSimpleCBOW(vocabSize, hiddenSize)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := trainer.InitTrainer(model, optimizer)

	trainer.Fit3D(co, matutil.At3D(te, 0), maxEpoch, batchSize)
	dist := trainer.GetWordDist()
	w2v := word.GetWord2VecFromDist(dist, id2w)
	for w, v := range w2v {
		fmt.Printf("=== %v ===\n", w)
		matutil.PrintMat(v)
	}
}
