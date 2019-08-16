<p align="center">
  <h3 align="center">gonnp</h3>
  <p align="center">Deep learning from scratch using Go. Specializes in natural language processing</p>
</p>

---
<img src="https://img.shields.io/badge/go-v1.12-blue.svg"/> [![CircleCI](https://circleci.com/gh/po3rin/gonnp.svg?style=shield&circle-token=d2ad1b26978ffeb0f6aa43b9a517ec7e5180d474)](https://circleci.com/gh/po3rin/gonnp) <a href="https://codeclimate.com/github/po3rin/gonnp/maintainability"><img src="https://api.codeclimate.com/v1/badges/a0e4c5e4c1c04fafb73a/maintainability" /></a> [![GolangCI](https://golangci.com/badges/github.com/po3rin/gonnp.svg)](https://golangci.com) [![codecov](https://codecov.io/gh/po3rin/gonnp/branch/master/graph/badge.svg)](https://codecov.io/gh/po3rin/gonnp) [![GoDoc](https://godoc.org/github.com/po3rin/gonnp?status.svg)](https://godoc.org/github.com/po3rin/gonnp)

## What

Package gonnp is the library of neural network components specialized in natural language processing in Go.　You can assemble a neural network with the necessary components.

## Dependencies

This component depends on ```gonum.org/v1/gonum/mat```
https://github.com/gonum/gonum/

## Components

The number of components will increase in the future.

### Layers

* Affine
* MatuMul
* Embedding
* EmbeddingDot
* Relu
* Sigmoid
* Softmax with Loss
* Sigmoid with Loss

### Optimizer

* SDG
* Adam

## Quick Start

### Word2Vec

```go
package main

import (
	"fmt"

	"github.com/po3rin/gonnp/matutils"
	"github.com/po3rin/gonnp/nn"
	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/trainer"
	"github.com/po3rin/gonnp/word"
)

func main() {
        hiddenSize := 5
        batchSize := 3
        maxEpoch := 1000

        // prepare one-hot matrix from text data.
        text := "You say goodbye and I say hello."
        corpus, w2id, id2w := word.PreProcess(text)
        vocabSize := len(w2id)
        contexts, target := word.CreateContextsAndTarget(corpus)
        te := word.ConvertOneHot(target, vocabSize)
        co := word.ConvertOneHot(contexts, vocabSize)

        // Inits model
        model := nn.InitSimpleCBOW(vocabSize, hiddenSize)
        // choses optimizer
        optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
        // inits trainer with model & optimizer.
        trainer := trainer.InitTrainer(model, optimizer)

        // training !!
        trainer.Fit3D(co, matutils.At3D(te, 0), maxEpoch, batchSize)

        // checks outputs
        dist := trainer.GetWordDist()
        w2v := word.GetWord2VecFromDist(dist, id2w)
        for w, v := range w2v {
                  fmt.Printf("=== %v ===\n", w)
                  matutils.PrintMat(v)
        }
}
```

outputs

```bash
=== goodbye ===
⎡ -0.983712641282964⎤
⎢ 0.9633828650811918⎥
⎢-0.7253396760955725⎥
⎢-0.9927919148802162⎥
⎣ 0.9868140369919183⎦
=== and ===
    .
    .
    .
```

### MNIST

```go
package main

import (
        "github.com/po3rin/gomnist"
        "github.com/po3rin/gonnp/nn"
        "github.com/po3rin/gonnp/optimizers"
        "github.com/po3rin/gonnp/trainer"
)

func main() {
        model := nn.NewTwoLayerNet(784, 100, 10)
        optimizer := optimizers.InitSDG(0.01)
        trainer := trainer.InitTrainer(model, optimizer, trainer.EvalInterval(20))

        // load MNIST data using github.com/po3rin/gomnist package
        l := gomnist.NewLoader("./../testdata", gomnist.OneHotLabel(true), gomnist.Normalization(true))
        mnist, _ := l.Load()

        trainer.Fit(mnist.TestData, mnist.TestLabels, 10, 100)
}
```

## TODO

Impliments Negative Sampling
Impliments RNN
