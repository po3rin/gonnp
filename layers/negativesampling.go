package layers

import (
	"github.com/po3rin/gonlp/entity"
	"gonum.org/v1/gonum/mat"
)

type EmbeddingDot struct {
	IDx   mat.Matrix
	Param entity.Param
	Grad  entity.Grad
}

// class EmbeddingDot:
//     def __init__(self, W):
//         self.embed = Embedding(W)
//         self.params = self.embed.params
//         self.grads = self.embed.grads
//         self.cache = None

//     def forward(self, h, idx):
//         target_W = self.embed.forward(idx)
//         out = np.sum(target_W * h, axis=1)

//         self.cache = (h, target_W)
//         return out

//     def backward(self, dout):
//         h, target_W = self.cache
//         dout = dout.reshape(dout.shape[0], 1)

//         dtarget_W = dout * h
//         self.embed.backward(dtarget_W)
//         dh = dout * target_W
//         return dh

// InitEmbeddingDotLayer inits Relu layer.
func InitEmbeddingDotLayer(weight mat.Matrix) *EmbeddingDot {
	return &EmbeddingDot{}
}

func (e *EmbeddingDot) Forward(x mat.Matrix) mat.Matrix {
	return nil
}

func (e *EmbeddingDot) Backward(x mat.Matrix) mat.Matrix {
	return nil
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
