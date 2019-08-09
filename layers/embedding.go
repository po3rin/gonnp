package layers

import (
	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/matutils"
	"gonum.org/v1/gonum/mat"
)

type Embedding struct {
	IDx   mat.Matrix
	Param entity.Param
	Grad  entity.Grad
}

// InitReluLayer inits Relu layer.
func InitEmbeddingLayer(weight mat.Matrix) *Embedding {
	r, c := weight.Dims()
	grad := mat.NewDense(r, c, nil)
	return &Embedding{
		Param: entity.Param{
			Weight: weight,
		},
		Grad: entity.Grad{
			Weight: grad,
		},
	}
}

func (e *Embedding) Forward(x mat.Matrix) mat.Matrix {
	e.IDx = x
	dout := matutils.ThinRowWithMat(e.Param.Weight, x)
	return dout
}

func (e *Embedding) Backward(x mat.Matrix) mat.Matrix {
	r, c := e.Param.Weight.Dims()
	gw := mat.NewDense(r, c, nil)

	d, ok := x.(*mat.Dense)
	if !ok {
		panic("gonlp: failed to transpose matrix to dense")
	}

	r, _ = e.IDx.Dims()
	for i := 0; i < r; i++ {
		id := int(e.IDx.At(i, 0))
		xv := d.RowView(i)
		wv := gw.RowView(id)

		add := mat.NewDense(c, 1, nil)
		add.Add(wv, xv)

		// TODO: refactoring...
		ar, ac := add.Dims()
		fs := make([]float64, 0, ar)
		for i := 0; i < ar; i++ {
			for j := 0; j < ac; j++ {
				fs = append(fs, add.At(i, j))
			}
		}
		gw.SetRow(id, fs)
	}
	e.Grad.Weight = gw
	return nil
}

func (e *Embedding) GetParam() entity.Param {
	return e.Param
}

func (e *Embedding) GetGrad() entity.Grad {
	return e.Grad
}

func (e *Embedding) SetParam(p entity.Param) {
	e.Param = p
}
