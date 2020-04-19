package xlayers

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

func (e *Embedding) Forward(out chan<- mat.Matrix, in ...<-chan mat.Matrix) {
	e.IDx = <-in[0]
	dout := matutil.ThinRowWithMat(e.Param.Weight, e.IDx)
	out <- dout
}

func (e *Embedding) Backward(out chan<- mat.Matrix, in ...<-chan mat.Matrix) {
	r, c := e.Param.Weight.Dims()
	gw := mat.NewDense(r, c, nil)

	x := <-in[0]

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
	out <- &mat.Dense{}
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
