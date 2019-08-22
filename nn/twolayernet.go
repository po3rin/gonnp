package nn

import (
	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

// TwoLayerNet has layer net config.
type TwoLayerNet struct {
	Layers    []Layer
	LossLayer LossLayer
}

// NewTwoLayerNet inits 2-layer-network.
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int) *TwoLayerNet {
	w1 := weightGenerator(inputSize, hiddenSize)
	w2 := weightGenerator(hiddenSize, outputSize)

	b1 := biasGenerator(hiddenSize, nil)
	b2 := biasGenerator(outputSize, nil)

	ls := []Layer{
		layers.InitAffineLayer(w1, b1),
		layers.InitReluLayer(),
		// layers.InitSigmoidLayer(),
		layers.InitAffineLayer(w2, b2),
	}

	return &TwoLayerNet{
		Layers:    ls,
		LossLayer: layers.InitSoftmaxWithLossLayer(),
	}
}

func (t *TwoLayerNet) Predict(x mat.Matrix) mat.Matrix {
	for i, l := range t.Layers {
		x = l.Forward(x)
		t.Layers[i] = l
	}
	return x
}

func (t *TwoLayerNet) Forward(teacher mat.Matrix, x ...mat.Matrix) float64 {
	m := x[0]
	for i, l := range t.Layers {
		m = l.Forward(m)
		t.Layers[i] = l
	}
	score := m
	loss := t.LossLayer.Forward(score, teacher)
	return loss
}

func (t *TwoLayerNet) Backward() mat.Matrix {
	dout := t.LossLayer.Backward()
	for i := len(t.Layers) - 1; i >= 0; i-- {
		dout = t.Layers[i].Backward(dout)
	}
	return dout
}

// GetParams gets params that layers have.
func (t *TwoLayerNet) GetParams() []entity.Param {
	params := make([]entity.Param, 0, len(t.Layers))
	for _, l := range t.Layers {
		// ignore if weight is empty.
		if l.GetParam().Weight == nil {
			continue
		}
		params = append(params, l.GetParam())
	}
	return params
}

// GetGrads gets gradient that layers have.
func (t *TwoLayerNet) GetGrads() []entity.Grad {
	grads := make([]entity.Grad, 0, len(t.Layers))
	for _, l := range t.Layers {
		// ignore if weight is empty.
		if l.GetGrad().Weight == nil {
			continue
		}
		grads = append(grads, l.GetGrad())
	}
	return grads
}

// UpdateParams updates lyaers params using TwoLayerMet's params.
func (t *TwoLayerNet) UpdateParams(params []entity.Param) {
	var i int
	for j, l := range t.Layers {
		p := l.GetParam()
		// ignore if weight is nil.
		if p.Weight == nil || p.Bias == nil {
			continue
		}
		l.SetParam(params[i])
		t.Layers[j] = l
		i++
	}
}
