// +build e2e

package e2e_test

import (
	"testing"

	"github.com/po3rin/gonlp/matutils"
	"github.com/po3rin/gonlp/nn"
	"github.com/po3rin/gonlp/optimizers"
	"github.com/po3rin/gonlp/trainer"
	"github.com/po3rin/gonlp/word"
)

func TestSimpleCBOW(t *testing.T) {
	hiddenSize := 5
	batchSize := 3
	maxEpoch := 1000

	text := "You say goodbye and I say hello."
	corpus, w2id, _ := word.PreProcess(text)

	vocabSize := len(w2id)
	contexts, target := word.CreateContextsAndTarget(corpus)

	te := word.ConvertOneHot(target, vocabSize)
	co := word.ConvertOneHot(contexts, vocabSize)

	model := nn.InitSimpleCBOW(vocabSize, hiddenSize)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := trainer.InitTrainer(model, optimizer)

	// TODO: impliments
	trainer.Fit3D(co, matutils.At3D(te, 0), maxEpoch, batchSize)
}
