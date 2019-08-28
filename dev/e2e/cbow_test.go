// +build e2e

package e2e_test

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/po3rin/gonnp/matutils"
	"github.com/po3rin/gonnp/nn"
	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/trainer"
	"github.com/po3rin/gonnp/word"
)

func TestCBOW(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	windowSize := 5
	hiddenSize := 100
	batchSize := 100
	maxEpoch := 1

	file, err := os.Open("../../testdata/golang.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	text, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatal(err)
	}
	corpus, w2id, id2w := word.PreProcess(string(text))
	vocabSize := len(w2id)

	contexts, target := word.CreateContextsAndTarget(corpus, windowSize)

	model := nn.InitCBOW(vocabSize, hiddenSize, windowSize, corpus)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := trainer.InitTrainer(model, optimizer)
	trainer.Fit(contexts, target, maxEpoch, batchSize)
	dist := trainer.GetWordDist()
	w2v := word.GetWord2VecFromDist(dist, id2w)
	for w, v := range w2v {
		fmt.Printf("=== %v ===\n", w)
		matutils.PrintMat(v)
	}
}
