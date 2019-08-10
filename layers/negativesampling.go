package layers

import (
	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/matutils"
	"gonum.org/v1/gonum/mat"
)

type cache struct {
	h       mat.Matrix
	targetW mat.Matrix
}

type EmbeddingDot struct {
	Embed *Embedding
	Param entity.Param
	Grad  entity.Grad
	cache cache
}

// InitEmbeddingDotLayer inits Relu layer.
func InitEmbeddingDotLayer(weight mat.Matrix) *EmbeddingDot {
	embed := InitEmbeddingLayer(weight)
	return &EmbeddingDot{
		Embed: embed,
		Param: embed.GetParam(),
		Grad:  embed.GetGrad(),
	}
}

func (e *EmbeddingDot) Forward(h mat.Matrix, idx mat.Matrix) mat.Matrix {
	targetW := e.Embed.Forward(idx)
	r, c := targetW.Dims()
	mul := mat.NewDense(r, c, nil)
	mul.MulElem(targetW, h)
	got := matutils.SumRow(mul)

	e.cache.h = h
	e.cache.targetW = targetW

	return got
}

func (e *EmbeddingDot) Backward(x mat.Matrix) mat.Matrix {
	h := e.cache.h
	targetW := e.cache.targetW

	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to transpose matrix to dense")
	}
	r, _ := d.Dims()
	dout := mat.NewDense(r, 1, d.RawMatrix().Data)

	r, c := dout.Dims()
	mul := mat.NewDense(r, c, nil)
	mul.MulElem(dout, h)

	_ = e.Embed.Backward(mul)

	mul.MulElem(dout, targetW)

	return mul
}

func (e *EmbeddingDot) GetParam() entity.Param {
	return e.Param
}

func (e *EmbeddingDot) GetGrad() entity.Grad {
	return e.Grad
}

func (e *EmbeddingDot) SetParam(p entity.Param) {
	e.Param = p
}
