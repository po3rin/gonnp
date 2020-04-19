package xlayers

import (
	"math"

	"github.com/po3rin/gonnp/matutil"
	"gonum.org/v1/gonum/mat"
)

// SigmoidWithLoss is layer for computing the multinomial logistic loss of the sigmoid of its inputs
type SigmoidWithLoss struct {
	X       mat.Matrix
	Teacher mat.Matrix
}

// InitSigmoidWithLossLayer inits sigmoid loss layer.
func InitSigmoidWithLossLayer() *SigmoidWithLoss {
	return &SigmoidWithLoss{}
}

// Forward for sigmoid loss layer.
// input fisrt argument is data, second is teacher data.
func (s *SigmoidWithLoss) Forward(out chan<- float64, in ...<-chan mat.Matrix) {
	x := <-in[0]
	x = sigmoid(x)
	s.X = x
	s.Teacher = <-in[1]

	y := mat.DenseCopyOf(x)
	f := func(i, j int, v float64) float64 {
		return 1 - v
	}
	y.Apply(f, y)
	out <- crossEntropyErr(matutil.JoinC(y, x), s.Teacher)
}

// Backward for sigmoid loss layer.
func (s *SigmoidWithLoss) Backward(out chan<- mat.Matrix) {
	batchSize, c := s.Teacher.Dims()
	f := func(i, j int, v float64) float64 {
		return (v - s.Teacher.At(i, j)) / float64(batchSize)
	}
	dx := mat.NewDense(batchSize, c, nil)
	dx.Apply(f, s.X)
	out <- dx
}

func sigmoid(x mat.Matrix) mat.Matrix {
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}

	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(sigmoid, x)
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
