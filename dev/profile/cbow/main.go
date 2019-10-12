package main

import (
	"fmt"

	"github.com/pkg/profile"
	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/models"
	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/testdata/ptb"
	"github.com/po3rin/gonnp/trainer"
	"github.com/po3rin/gonnp/word"
)

func main() {
	defer profile.Start(profile.ProfilePath(".")).Stop()

	windowSize := 5
	hiddenSize := 100
	batchSize := 100
	maxEpoch := 1

	corpus, w2id, id2w := ptb.LoadData("testdata", "train")
	vocabSize := len(w2id)

	contexts, target := word.CreateContextsAndTarget(corpus, windowSize)

	model := models.InitCBOW(vocabSize, hiddenSize, windowSize, corpus)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := trainer.InitTrainer(model, optimizer, trainer.EvalInterval(1))

	trainer.Fit(contexts, target, maxEpoch, batchSize)

	dist := trainer.GetWordDist()
	w2v := word.GetWord2VecFromDist(dist, id2w)
	for w, v := range w2v {
		fmt.Printf("=== %v ===\n", w)
		matutil.PrintMat(v)
	}
}
