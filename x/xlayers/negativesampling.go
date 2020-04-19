package xlayers

import (
	"time"

	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/params"
	"github.com/po3rin/gonnp/word"
	"gonum.org/v1/gonum/mat"
)

type cache struct {
	h       mat.Matrix
	targetW mat.Matrix
}

type EmbeddingDot struct {
	Embed *Embedding
	cache cache
}

// InitEmbeddingDotLayer inits Relu layer.
func InitEmbeddingDotLayer(weight mat.Matrix) *EmbeddingDot {
	embed := InitEmbeddingLayer(weight)
	return &EmbeddingDot{
		Embed: embed,
	}
}

func (e *EmbeddingDot) Forward(out chan<- mat.Matrix, in ...<-chan mat.Matrix) {
	tw := make(chan mat.Matrix, 1)
	e.Embed.Forward(tw, in[1])
	targetW := <-tw
	r, c := targetW.Dims()
	mul := mat.NewDense(r, c, nil)
	h := <-in[0]

	mul.MulElem(targetW, h)
	got := matutil.SumRow(mul)

	out <- got

	e.cache.h = h
	e.cache.targetW = targetW
}

func (e *EmbeddingDot) Backward(out chan<- mat.Matrix, in ...<-chan mat.Matrix) {
	h := e.cache.h
	targetW := e.cache.targetW

	x := <-in[0]
	r, _ := x.Dims()
	d := mat.DenseCopyOf(x)

	dout := mat.NewDense(r, 1, d.RawMatrix().Data)
	dv := mat.NewVecDense(r, dout.RawMatrix().Data)
	x = matutil.MulMatVec(h, dv)

	o := make(chan mat.Matrix, 1)
	i := make(chan mat.Matrix, 1)

	go e.Embed.Backward(o, i)

	i <- x
	<-o

	dh := matutil.MulMatVec(targetW, dv)
	out <- dh
}

func (e *EmbeddingDot) GetParam() params.Param {
	return e.Embed.GetParam()
}

func (e *EmbeddingDot) GetGrad() params.Grad {
	return e.Embed.GetGrad()
}

func (e *EmbeddingDot) SetParam(p params.Param) {
	e.Embed.SetParam(p)
}

type Sampler interface {
	GetNegativeSample(target mat.Vector) mat.Matrix
}

// NegativeSamplingLoss is layer for negative sampling.
type NegativeSamplingLoss struct {
	SampleSize     int
	EmbedDotLayers []*EmbeddingDot
	LossLayers     []*SigmoidWithLoss
	Sampler        Sampler
}

// InitNegativeSamplingLoss inits NegativeSamplingLoss.
func InitNegativeSamplingLoss(
	weight mat.Matrix,
	corpus word.Corpus,
	sampler Sampler,
	sampleSize int,
) *NegativeSamplingLoss {
	lossLayers := make([]*SigmoidWithLoss, 0, sampleSize+1)
	embedDotLayers := make([]*EmbeddingDot, 0, sampleSize+1)

	for i := 0; i < sampleSize+1; i++ {
		lossLayers = append(lossLayers, InitSigmoidWithLossLayer())
		embedDotLayers = append(embedDotLayers, InitEmbeddingDotLayer(weight))
	}

	return &NegativeSamplingLoss{
		SampleSize:     sampleSize,
		EmbedDotLayers: embedDotLayers,
		LossLayers:     lossLayers,
		Sampler:        sampler,
	}
}

// Forward calicurates loss with negative sampling.
func (n *NegativeSamplingLoss) Forward(out chan<- float64, in ...<-chan mat.Matrix) {
	target := <-in[1]
	batchSize, _ := target.Dims()
	td := mat.DenseCopyOf(target)
	vt := td.ColView(0)

	negativeSample := n.Sampler.GetNegativeSample(vt)
	nsd := mat.DenseCopyOf(negativeSample)

	s := make([]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		s[i] = 1
	}

	// correct forward
	scoreC := make(chan mat.Matrix, 1)
	hC := make(chan mat.Matrix, 1)
	targetC := make(chan mat.Matrix, 1)
	lossC := make(chan float64, 1)
	correctLabelC := make(chan mat.Matrix, 1)

	go n.EmbedDotLayers[0].Forward(scoreC, hC, targetC)
	go n.LossLayers[0].Forward(lossC, scoreC, correctLabelC)

	h := <-in[0]

	hC <- h
	targetC <- target
	correctLabelC <- mat.NewDense(batchSize, 1, s)

	loss := <-lossC

	// negative forward
	lossStream := make(chan float64, n.SampleSize)
	negativeLabel := mat.NewDense(batchSize, 1, nil)

	go func() {
		defer close(lossStream)
		for i := 0; i < n.SampleSize; i++ {

			time.Sleep(3 * time.Second)

			nt := nsd.ColView(i)

			scoreC := make(chan mat.Matrix, 1)
			hC := make(chan mat.Matrix, 1)
			negativeTargetC := make(chan mat.Matrix, 1)
			negativeLabelC := make(chan mat.Matrix, 1)
			lossC := make(chan float64, 1)

			go n.EmbedDotLayers[i+1].Forward(scoreC, hC, negativeTargetC)
			go n.LossLayers[i+1].Forward(lossC, scoreC, negativeLabelC)

			hC <- h
			negativeTargetC <- nt
			negativeLabelC <- negativeLabel
			lossStream <- <-lossC
		}
	}()

	for negativeLoss := range lossStream {
		loss += negativeLoss
	}

	out <- loss
}

func (n *NegativeSamplingLoss) Backward(out chan<- mat.Matrix) {
	var dh *mat.Dense
	for i, l := range n.LossLayers {
		dscore := make(chan mat.Matrix, 1)
		r := make(chan mat.Matrix, 1)

		go l.Backward(dscore)
		go n.EmbedDotLayers[i].Backward(r, dscore)

		m := <-r
		if dh == nil {
			dr, dc := m.Dims()
			dh = mat.NewDense(dr, dc, nil)
		}
		dh.Add(dh, m)
		n.LossLayers[i] = l
	}
	out <- dh
}

// GetParams gets params that layers have.
func (n *NegativeSamplingLoss) GetParams() []params.Param {
	params := make([]params.Param, 0, len(n.EmbedDotLayers))
	for _, l := range n.EmbedDotLayers {
		// ignore if weight is empty.
		if l.GetParam().Weight == nil {
			continue
		}
		params = append(params, l.GetParam())
	}
	return params
}

// GetGrads gets gradient that layers have.
func (n *NegativeSamplingLoss) GetGrads() []params.Grad {
	grads := make([]params.Grad, 0, len(n.EmbedDotLayers))
	for _, l := range n.EmbedDotLayers {
		// ignore if weight is empty.
		if l.GetGrad().Weight == nil {
			continue
		}
		grads = append(grads, l.GetGrad())
	}
	return grads
}

// UpdateParams updates lyaers params using args.
func (n *NegativeSamplingLoss) UpdateParams(params []params.Param) {
	var i int
	for j, l := range n.EmbedDotLayers {
		p := l.GetParam()
		// ignore if weight is nil.
		if p.Weight == nil {
			continue
		}
		l.SetParam(params[0])
		n.EmbedDotLayers[j] = l
		i++
	}
}
