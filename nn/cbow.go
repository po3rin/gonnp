package nn

import (
	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/layers"
	"github.com/po3rin/gonnp/word"
	"gonum.org/v1/gonum/mat"
)

type CBOW struct {
	Layers    []layers.Layer
	LossLayer layers.LossLayerWithParams
}

func InitCBOW(vocabSize, hiddenSize, windowSize int, corpus word.Corpus) *CBOW {
	sampleSize := 5

	w1 := weightGenerator(vocabSize, hiddenSize)
	w2 := weightGenerator(vocabSize, hiddenSize)

	ls := []layers.Layer{}
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
	var h *mat.Dense
	for i, l := range s.Layers {
		r := l.Forward(d.Slice(0, dr, i, i+1))
		if h == nil {
			rr, rc := r.Dims()
			h = mat.NewDense(rr, rc, nil)
		}
		h.Add(h, r)
	}

	h.Scale(float64(1/len(s.Layers)), h)
	return s.LossLayer.Forward(h, target)
}

func (s *CBOW) Backward() mat.Matrix {
	dout := s.LossLayer.Backward()
	d := mat.DenseCopyOf(dout)
	d.Scale(float64(1/len(s.Layers)), d)
	for _, l := range s.Layers {
		l.Backward(d)
	}
	return nil
}

// GetParams gets params that layers have.
func (s *CBOW) GetParams() []entity.Param {
	params := make([]entity.Param, 0, len(s.Layers))
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
func (s *CBOW) GetGrads() []entity.Grad {
	grads := make([]entity.Grad, 0, len(s.Layers))
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
func (s *CBOW) UpdateParams(params []entity.Param) {
	for j, l := range s.Layers {
		p := l.GetParam()
		// ignore if weight is nil.
		if p.Weight == nil {
			continue
		}
		l.SetParam(params[0])
		s.Layers[j] = l
	}

	s.LossLayer.UpdateParams([]entity.Param{params[1]})
}
