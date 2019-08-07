package nn

import (
	"github.com/po3rin/gonlp/layers"
	"gonum.org/v1/gonum/mat"
)

type SimpleCbow struct {
	Layers    []layers.Layer
	LossLayer layers.LossLayer
}

func NewSimpleCbow(vocabSize, hiddenSize int) *SimpleCbow {
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

func (s *SimpleCbow) Forward(contexts, target mat.Matrix) float64 {
	return 0
}

func (s *SimpleCbow) Backward() mat.Matrix {
	return nil
}
