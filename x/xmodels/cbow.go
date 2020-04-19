package xmodels

import (
	"github.com/po3rin/gonnp/layers"
	"github.com/po3rin/gonnp/params"
	"github.com/po3rin/gonnp/word"
	"github.com/po3rin/gonnp/x/xlayers"
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
		ls = append(ls, xlayers.InitEmbeddingLayer(w1))
	}

	sampler := layers.InitUnigraSampler(corpus, 0.75, sampleSize)
	return &CBOW{
		Layers:    ls,
		LossLayer: xlayers.InitNegativeSamplingLoss(w2, corpus, sampler, sampleSize),
	}
}

// Forward is CBOW models forward.
// first arg is data, secound is target.
func (s *CBOW) Forward(out chan<- float64, in ...<-chan mat.Matrix) {
	matrixStream := make(chan mat.Matrix, len(s.Layers))

	context := <-in[1]
	d := mat.DenseCopyOf(context)
	dr, _ := d.Dims()

	go func() {
		defer close(matrixStream)
		for i, l := range s.Layers {
			out := make(chan mat.Matrix, 1)
			in := make(chan mat.Matrix, 1)

			go l.Forward(out, in)

			in <- d.Slice(0, dr, i, i+1)
			matrixStream <- <-out
		}
	}()

	h := mat.NewDense(dr, dr, nil)
	for m := range matrixStream {
		h.Add(h, m)
	}

	h.Scale(1/float64(len(s.Layers)), h)

	hc := make(chan mat.Matrix, 1)
	loss := make(chan float64, 1)
	go s.LossLayer.Forward(loss, hc, in[0])

	hc <- h
	out <- <-loss
}

func (s *CBOW) Backward(out chan<- mat.Matrix) {
	dout := make(chan mat.Matrix)

	go s.LossLayer.Backward(dout)

	d := mat.DenseCopyOf(<-dout)

	d.Scale(1/float64(len(s.Layers)), d)
	for i, l := range s.Layers {
		in := make(chan mat.Matrix, 1)
		out := make(chan mat.Matrix, 1)

		go l.Backward(out, in)

		in <- d
		<-out

		s.Layers[i] = l
	}

	out <- &mat.Dense{}
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
