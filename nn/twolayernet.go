package nn

import (
	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/layers"
	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

// NodeSize has node size (input, hidden, output)
type NodeSize struct {
	Input  int
	Hideen int
	Output int
}

// TowLayerNet has layer net config.
type TowLayerNet struct {
	NodeSize  NodeSize
	Layers    []layers.Layer
	LossLayer layers.LossLayer
	Params    []entity.Param
	Grads     []entity.Grad
}

// NewTwoLayerNet inits 2-layer-network.
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int) *TowLayerNet {
	w1 := matutils.NewRandMatrixWithSND(inputSize, hiddenSize)
	w2 := matutils.NewRandMatrixWithSND(hiddenSize, outputSize)

	b1 := mat.NewDense(1, hiddenSize, nil)
	b2 := mat.NewDense(1, outputSize, nil)

	ls := []layers.Layer{
		layers.InitAffineLayer(w1, b1),
		layers.InitSigmoidLayer(),
		layers.InitAffineLayer(w2, b2),
	}

	var (
		params []entity.Param
		grads  []entity.Grad
	)
	for _, l := range ls {
		p := l.GetParam()
		g := l.GetGrad()
		params = append(params, p)
		grads = append(grads, g)
	}

	return &TowLayerNet{
		NodeSize: NodeSize{
			Input:  inputSize,
			Hideen: hiddenSize,
			Output: outputSize,
		},
		Layers:    ls,
		LossLayer: layers.InitSoftmaxWithLossLayer(),
		Params:    params,
		Grads:     grads,
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
	if x == nil {
		// TODO: set correct size
		x = mat.NewDense(1, 1, []float64{1})
	}
	dout := t.LossLayer.Backward(x)
	for i := len(t.Layers) - 1; i >= 0; i-- {
		dout = t.Layers[i].Backward(dout)
	}
	return dout
}

// GetParams gets layer net params.
func (t *TowLayerNet) GetParams() []entity.Param {
	return t.Params
}

// GetGrads gets layer net gradient.
func (t *TowLayerNet) GetGrads() []entity.Grad {
	return t.Grads
}

// SetParams updates params.
func (t *TowLayerNet) SetParams(params []entity.Param) {
	t.Params = params
}
