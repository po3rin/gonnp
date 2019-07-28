package twolayernet

import (
	"github.com/po3rin/gonlp/layers"
	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

type NodeSize struct {
	Input  int
	Hideen int
	Output int
}

type TowLayerNet struct {
	NodeSize  NodeSize
	Layers    []layers.Layer
	LossLayer layers.LossLayer
}

// NewTwoKayerNet inits 2-layer-network.
func (t *TowLayerNet) NewTwoKayerNet(inputSize, hiddenSize, outputSize int) *TowLayerNet {
	w1 := matutils.NewRandMatrixWithSND(inputSize, hiddenSize)
	w2 := matutils.NewRandMatrixWithSND(hiddenSize, outputSize)

	b1 := mat.NewDense(1, hiddenSize, nil)
	b2 := mat.NewDense(1, outputSize, nil)

	return &TowLayerNet{
		NodeSize: NodeSize{
			Input:  inputSize,
			Hideen: hiddenSize,
			Output: outputSize,
		},
		Layers: []layers.Layer{
			layers.InitAffineLayer(w1, b1),
			layers.InitSigmoidLayer(),
			layers.InitAffineLayer(w2, b2),
		},
		LossLayer: layers.InitSoftmaxWithLossLayer(),
	}
}

func (t *TowLayerNet) Predict(x mat.Matrix) mat.Matrix {
	for _, l := range t.Layers {
		x = l.Forward(x)
	}
	return x
}

func (t *TowLayerNet) Forward(x mat.Matrix, teacher mat.Matrix) float64 {
	score := t.Predict(x)
	loss := t.LossLayer.Forward(score, teacher)
	return loss
}

func (t *TowLayerNet) Backward(x mat.Matrix) mat.Matrix {
	dout := t.LossLayer.Backward(x)
	for i := len(t.Layers) - 1; i >= 0; i-- {
		dout = t.Layers[i].Backward(dout)
	}
	return dout
}
