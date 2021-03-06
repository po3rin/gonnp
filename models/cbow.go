package models

import (
	"github.com/po3rin/gonnp/layers"
	"github.com/po3rin/gonnp/params"
	"github.com/po3rin/gonnp/word"
	"gonum.org/v1/gonum/mat"
)

type CBOW struct {
	Layers    []Layer
	LossLayer LossLayerWithParams
}

func InitCBOW(vocabSize, hiddenSize, windowSize int, corpus word.Corpus) *CBOW {
	sampleSize := 5

	w1 := weightGenerator(vocabSize, hiddenSize)
	w2 := weightGenerator(vocabSize, hiddenSize)

	ls := []Layer{}
	for i := 0; i < windowSize*2; i++ {
		ls = append(ls, layers.InitEmbeddingLayer(w1))
	}

	sampler := layers.InitUnigraSampler(corpus, 0.75, sampleSize)
	return &CBOW{
		Layers:    ls,
		LossLayer: layers.InitNegativeSamplingLoss(w2, corpus, sampler, sampleSize),
	}
}

func (s *CBOW) Forward(target mat.Matrix, contexts ...mat.Matrix) float64 {
	d := mat.DenseCopyOf(contexts[0])
	dr, _ := d.Dims()
	h := mat.NewDense(dr, dr, nil)

	for i, l := range s.Layers {
		r := l.Forward(d.Slice(0, dr, i, i+1))
		h.Add(h, r)
	}

	h.Scale(1/float64(len(s.Layers)), h)
	return s.LossLayer.Forward(h, target)
}

func (s *CBOW) Backward() mat.Matrix {
	dout := s.LossLayer.Backward()
	d := mat.DenseCopyOf(dout)

	d.Scale(1/float64(len(s.Layers)), d)
	for i, l := range s.Layers {
		l.Backward(d)
		s.Layers[i] = l
	}
	return nil
}

// GetParams gets params that layers have.
func (s *CBOW) GetParams() []params.Param {
	params := make([]params.Param, 0, len(s.Layers))
	for _, l := range s.Layers {
		// ignore if weight is empty.
		if l.GetParam().Weight == nil {
			continue
		}
		params = append(params, l.GetParam())
	}
	params = append(params, s.LossLayer.GetParams()...)
	return params
}

// GetGrads gets gradient that layers have.
func (s *CBOW) GetGrads() []params.Grad {
	grads := make([]params.Grad, 0, len(s.Layers))
	for _, l := range s.Layers {
		// ignore if weight is empty.
		if l.GetGrad().Weight == nil {
			continue
		}
		grads = append(grads, l.GetGrad())
	}
	grads = append(grads, s.LossLayer.GetGrads()...)
	return grads
}

// UpdateParams updates lyaers params using TwoLayerMet's params.
func (s *CBOW) UpdateParams(ps []params.Param) {
	for j, l := range s.Layers {
		p := l.GetParam()
		// ignore if weight is nil.
		if p.Weight == nil {
			continue
		}
		l.SetParam(ps[0])
		s.Layers[j] = l
	}

	s.LossLayer.UpdateParams([]params.Param{ps[1]})
}
