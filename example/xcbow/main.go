package main

import (
	"log"
	"math/rand"
	"time"

	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/store"
	"github.com/po3rin/gonnp/testdata/ptb"
	"github.com/po3rin/gonnp/word"
	"github.com/po3rin/gonnp/x/xmodels"
	"github.com/po3rin/gonnp/x/xtrainer"
)

// train Word2Vec.
func main() {
	rand.Seed(time.Now().UnixNano())

	windowSize := 5
	hiddenSize := 100
	batchSize := 100
	maxEpoch := 10

	corpus, w2id, id2w := ptb.LoadData("testdata", "train")
	vocabSize := len(w2id)

	contexts, target := word.CreateContextsAndTarget(corpus, windowSize)

	model := xmodels.InitCBOW(vocabSize, hiddenSize, windowSize, corpus)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := xtrainer.InitTrainer(model, optimizer, xtrainer.EvalInterval(20))

	trainer.Fit(contexts, target, maxEpoch, batchSize)

	dist := trainer.GetWordDist()
	err := store.NewCBOWEncoder(w2id, id2w, dist).Encode("cbow.gob")
	if err != nil {
		log.Fatal(err)
	}
}