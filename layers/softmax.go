package layers

import (
	"math"

	"github.com/po3rin/gonnp/matutil"
	"gonum.org/v1/gonum/mat"
)

// SoftmaxWithLoss is layer for computing the multinomial logistic loss of the softmax of its inputs
type SoftmaxWithLoss struct {
	X       mat.Matrix
	Teacher mat.Matrix
}

// InitSoftmaxWithLossLayer inits softmax layer.
func InitSoftmaxWithLossLayer() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

// Forward for softmax layer.
func (s *SoftmaxWithLoss) Forward(x mat.Matrix, teacher mat.Matrix) float64 {
	s.X = softmax(x)
	s.Teacher = teacher
	return crossEntropyErr(s.X, teacher)
}

// Backward for softmax layer.
func (s *SoftmaxWithLoss) Backward() mat.Matrix {
	batchSize, c := s.Teacher.Dims()
	f := func(i, j int, v float64) float64 {
		return (v - s.Teacher.At(i, j)) / float64(batchSize)
	}
	dx := mat.NewDense(batchSize, c, nil)
	dx.Apply(f, s.X)
	return dx
}

func softmax(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	if r == 1 || c == 1 {
		max := mat.Max(x)
		f := func(i, j int, v float64) float64 {
			return math.Exp(v - max)
		}
		var result mat.Dense
		result.Apply(f, x)
		sum := mat.Sum(&result)
		result.Scale(1/sum, &result)
		return &result
	}
	d := mat.NewDense(r, c, nil)
	m := matutil.SubMatVec(x, matutil.Mat2VecDenseWithColMax(x))
	f := func(i, j int, v float64) float64 {
		return math.Exp(v)
	}
	d.Apply(f, m)
	v := matutil.SumRow(d)
	result := matutil.DivMatVec(d, v)
	return result
}

// CrossEntropyErr measures the performance of a classification model whose output is a probability value between 0 and 1.
func crossEntropyErr(data mat.Matrix, teacher mat.Matrix) float64 {
	d := mat.DenseCopyOf(data)
	t := mat.DenseCopyOf(teacher)

	batchSize, dc := d.Dims()
	tr, tc := t.Dims()

	if batchSize == 1 {
		d = mat.NewDense(1, batchSize*dc, d.RawMatrix().Data)
		t = mat.NewDense(1, tr*tc, t.RawMatrix().Data)
	}

	if batchSize == tr && dc == tc {
		oh := matutil.OneHotVec2Index(t)
		t = mat.DenseCopyOf(oh)
	}

	tr, tc = t.Dims()
	if tr != 1 {
		t = mat.NewDense(1, tr*tc, t.RawMatrix().Data)
	}

	// d[np.arange(batch_size), t]
	fs := t.RawMatrix().Data
	ints := make([]int, len(fs))
	for i, v := range fs {
		ints[i] = int(v)
	}
	tr, tc = t.Dims()
	var sd mat.Matrix
	if tc == 1 {
		sd = matutil.SetColToRow(d, ints)
	} else if tr == 1 {
		sd = matutil.ExtractFromEachRows(d, ints)
	}

	crossEnrtopy := func(i, j int, v float64) float64 {
		return math.Log(v + 1e-7)
	}

	sdd := mat.DenseCopyOf(sd)
	sdd.Apply(crossEnrtopy, sd)
	return -mat.Sum(sdd) / float64(batchSize)
}
