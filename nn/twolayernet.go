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
	Hidden int
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

	b1 := mat.NewVecDense(hiddenSize, nil)
	b2 := mat.NewVecDense(outputSize, nil)

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
		if p.Weight == nil || p.Bias == nil {
			continue
		}
		params = append(params, p)
	}

	return &TowLayerNet{
		NodeSize: NodeSize{
			Input:  inputSize,
			Hidden: hiddenSize,
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

func (t *TowLayerNet) Backward() mat.Matrix {
	dout := t.LossLayer.Backward()
	for i := len(t.Layers) - 1; i >= 0; i-- {
		dout = t.Layers[i].Backward(dout)
	}
	return dout
}

// GetParams gets params that layers have.
func (t *TowLayerNet) GetParams() []entity.Param {
	params := make([]entity.Param, 0, len(t.Layers))
	for _, l := range t.Layers {
		if l.GetParam().Weight == nil {
			continue
		}
		params = append(params, l.GetParam())
	}
	return params
}

// GetGrads gets gradient that layers have.
func (t *TowLayerNet) GetGrads() []entity.Grad {
	grads := make([]entity.Grad, 0, len(t.Layers))
	for _, l := range t.Layers {
		if l.GetGrad().Weight == nil {
			continue
		}
		grads = append(grads, l.GetGrad())
	}
	return grads
}

// UpdateParams updates lyaers params using TwoLayerMet's params.
func (t *TowLayerNet) UpdateParams(params []entity.Param) {
	var i int
	for _, l := range t.Layers {
		p := l.GetParam()
		if p.Weight == nil || p.Bias == nil {
			continue
		}
		l.SetParam(params[i])
		i++
	}
}
