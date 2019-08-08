package nn

import (
	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/layers"
	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

type SimpleCbow struct {
	Layers    []layers.Layer
	LossLayer layers.LossLayer
}

func InitSimpleCBOW(vocabSize, hiddenSize int) *SimpleCbow {
	w1 := weightGenerator(vocabSize, hiddenSize)
	w2 := weightGenerator(hiddenSize, vocabSize)

	ls := []layers.Layer{
		layers.InitMatMulLayer(w1),
		layers.InitMatMulLayer(w1),
		layers.InitMatMulLayer(w2),
	}

	return &SimpleCbow{
		Layers:    ls,
		LossLayer: layers.InitSoftmaxWithLossLayer(),
	}
}

func (s *SimpleCbow) Forward(target mat.Matrix, contexts ...mat.Matrix) float64 {
	a0 := matutils.At3D(contexts, 0)
	a1 := matutils.At3D(contexts, 1)

	h0 := s.Layers[0].Forward(a0)
	h1 := s.Layers[1].Forward(a1)

	d0, ok := h0.(*mat.Dense)
	if !ok {
		panic("gonlp: failed to transpose matrix to dense")
	}
	d1, ok := h1.(*mat.Dense)
	if !ok {
		panic("gonlp: failed to transpose matrix to dense")
	}
	d0.Add(d0, d1)
	d0.Scale(0.5, d0)
	score := s.Layers[2].Forward(d0)
	return s.LossLayer.Forward(score, target)
}

func (s *SimpleCbow) Backward() mat.Matrix {
	ds := s.LossLayer.Backward()
	da := s.Layers[2].Backward(ds)

	d, ok := da.(*mat.Dense)
	if !ok {
		panic("gonlp: failed to transpose matrix to dense")
	}

	d.Scale(0.5, d)

	// fmt.Println("===============")
	// matutils.PrintMat(d)

	_ = s.Layers[0].Backward(d)
	_ = s.Layers[1].Backward(d)
	return nil
}

// GetParams gets params that layers have.
func (s *SimpleCbow) GetParams() []entity.Param {
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
func (s *SimpleCbow) GetGrads() []entity.Grad {
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
func (s *SimpleCbow) UpdateParams(params []entity.Param) {
	s.Layers[0].SetParam(params[0])
	s.Layers[1].SetParam(params[0])
	s.Layers[2].SetParam(params[1])
}
