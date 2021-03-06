package layers

import (
	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/params"
	"gonum.org/v1/gonum/mat"
)

// Embedding has Embedding layer param.
type Embedding struct {
	IDx   mat.Matrix
	Param params.Param
	Grad  params.Grad
}

// InitEmbeddingLayer inits Embedding layer.
func InitEmbeddingLayer(weight mat.Matrix) *Embedding {
	r, c := weight.Dims()
	grad := mat.NewDense(r, c, nil)
	return &Embedding{
		Param: params.Param{
			Weight: weight,
		},
		Grad: params.Grad{
			Weight: grad,
		},
	}
}

func (e *Embedding) Forward(x mat.Matrix) mat.Matrix {
	e.IDx = x
	dout := matutil.ThinRowWithMat(e.Param.Weight, x)
	return dout
}

func (e *Embedding) Backward(x mat.Matrix) mat.Matrix {
	r, c := e.Param.Weight.Dims()
	gw := mat.NewDense(r, c, nil)

	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to gonnp: not yet supported type matrix to dense")
	}

	r, _ = e.IDx.Dims()
	for i := 0; i < r; i++ {
		id := int(e.IDx.At(i, 0))
		xv := d.RowView(i)
		wv := gw.RowView(id)

		add := mat.NewDense(c, 1, nil)
		add.Add(wv, xv)
		gw.SetRow(id, add.RawMatrix().Data)
	}
	e.Grad.Weight = gw
	return nil
}

func (e *Embedding) GetParam() params.Param {
	return e.Param
}

func (e *Embedding) GetGrad() params.Grad {
	return e.Grad
}

func (e *Embedding) SetParam(p params.Param) {
	e.Param = p
}

// TimeEmbedding run embedding in bulk.
type TimeEmbedding struct {
	Param  params.Param
	Grad   params.Grad
	Layers []*Embedding
}

// InitTimeEmbeddingLayer inits TimeEmbedding layer.
func InitTimeEmbeddingLayer(weight mat.Matrix) *TimeEmbedding {
	return &TimeEmbedding{
		Param: params.Param{
			Weight: weight,
		},
	}
}

func (t *TimeEmbedding) Forward(xs mat.Matrix) []mat.Matrix {
	N, T := xs.Dims()
	_, D := t.Param.Weight.Dims()

	out := matutil.New3D(N, T, D)
	layers := make([]*Embedding, T)

	var md mat.Dense
	md.CloneFrom(xs)

	for i := 0; i < T; i++ {
		l := InitEmbeddingLayer(t.Param.Weight)
		in := md.ColView(i)
		o := l.Forward(in)
		matutil.Set3D(out, o, i)
		layers[i] = l
	}

	t.Layers = layers
	return out
}

func (t *TimeEmbedding) Backward(dout []mat.Matrix) mat.Matrix {
	T, _ := dout[0].Dims()

	r, c := t.Param.Weight.Dims()
	grad := mat.NewDense(r, c, nil)
	for i := 0; i < T; i++ {
		l := t.Layers[i]
		l.Backward(matutil.At3D(dout, i))
		grad.Add(grad, l.GetGrad().Weight)
	}

	t.Grad.Weight = grad
	return nil
}
