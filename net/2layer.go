package net

import (
	"github.com/po3rin/gonlp/layers"
	"gonum.org/v1/gonum/mat"
)

type TowLayerNet struct {
	Layers []layers.Layer
}

func (t *TowLayerNet) Predict() mat.Matrix {
	var x mat.Matrix
	for _, l := range t.Layers {
		x = l.Forward(x)
	}
	return x
}

func (t *TowLayerNet) Init() {

}
