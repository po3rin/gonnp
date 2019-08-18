package nn

import (
	"fmt"

	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/layers"
	"github.com/po3rin/gonnp/matutils"
	"github.com/po3rin/gonnp/word"
	"gonum.org/v1/gonum/mat"
)

type Cbow struct {
	Layers    []layers.Layer
	LossLayer layers.LossLayer
}

func InitCBOW(vocabSize, hiddenSize, windowSize int, corpus word.Corpus) *Cbow {
	sampleSize := 3

	w1 := weightGenerator(vocabSize, hiddenSize)
	w2 := weightGenerator(vocabSize, hiddenSize)

	ls := []layers.Layer{}
	for i := 0; i < windowSize*2; i++ {
		ls = append(ls, layers.InitEmbeddingLayer(w1))
	}

	sampler := layers.InitUnigraSampler(corpus, 0.75, sampleSize)
	return &Cbow{
		Layers:    ls,
		LossLayer: layers.InitNegativeSamplingLoss(w2, corpus, sampler, sampleSize),
	}
}

func (s *Cbow) Forward(target mat.Matrix, contexts ...mat.Matrix) float64 {
	var h *mat.Dense
	for i, l := range s.Layers {
		r := l.Forward(matutils.At3D(contexts, i))
		h.Add(h, r)
	}

	h.Scale(float64(1/len(s.Layers)), h)
	return s.LossLayer.Forward(h, target)
}

func (s *Cbow) Backward() mat.Matrix {
	dout := s.LossLayer.Backward()
	d := mat.DenseCopyOf(dout)
	d.Scale(float64(1/len(s.Layers)), d)
	for _, l := range s.Layers {
		l.Backward(d)
	}
	return nil
}

// GetParams gets params that layers have.
func (s *Cbow) GetParams() []entity.Param {
	params := make([]entity.Param, 0, len(s.Layers))
	for _, l := range s.Layers {
		// ignore if weight is empty.
		if l.GetParam().Weight == nil {
			continue
		}
		params = append(params, l.GetParam())
	}
	return params
}

// GetGrads gets gradient that layers have.
func (s *Cbow) GetGrads() []entity.Grad {
	grads := make([]entity.Grad, 0, len(s.Layers))
	for _, l := range s.Layers {
		// ignore if weight is empty.
		if l.GetGrad().Weight == nil {
			continue
		}
		grads = append(grads, l.GetGrad())
	}
	return grads
}

// UpdateParams updates lyaers params using TwoLayerMet's params.
func (s *Cbow) UpdateParams(params []entity.Param) {
	// TODO: impliments
	fmt.Println(len(params))
}
