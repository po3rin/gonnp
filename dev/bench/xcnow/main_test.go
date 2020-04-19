// +build !e2e

package main

import (
	"io/ioutil"
	"log"
	"os"
	"testing"

	"github.com/po3rin/gonnp/models"
	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/trainer"
	"github.com/po3rin/gonnp/word"
	"github.com/po3rin/gonnp/x/xmodels"
	"github.com/po3rin/gonnp/x/xtrainer"
)

var (
	windowSize = 5
	hiddenSize = 100
	batchSize  = 100
	maxEpoch   = 1
)

func BenchmarkCbow(b *testing.B) {
	b.ReportAllocs()

	file, err := os.Open("../../../testdata/golang.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	text, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatal(err)
	}
	corpus, w2id, _ := word.PreProcess(string(text))
	vocabSize := len(w2id)

	contexts, target := word.CreateContextsAndTarget(corpus, windowSize)

	model := models.InitCBOW(vocabSize, hiddenSize, windowSize, corpus)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := trainer.InitTrainer(model, optimizer, trainer.EvalInterval(1000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainer.Fit(contexts, target, maxEpoch, batchSize)
	}
}

func BenchmarkXCbow(b *testing.B) {
	b.ReportAllocs()

	file, err := os.Open("../../../testdata/golang.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	text, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatal(err)
	}
	corpus, w2id, _ := word.PreProcess(string(text))
	vocabSize := len(w2id)
	contexts, target := word.CreateContextsAndTarget(corpus, windowSize)

	model := xmodels.InitCBOW(vocabSize, hiddenSize, windowSize, corpus)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := xtrainer.InitTrainer(model, optimizer, xtrainer.EvalInterval(1000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainer.Fit(contexts, target, maxEpoch, batchSize)
	}
}
