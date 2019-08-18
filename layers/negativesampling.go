package layers

import (
	"math"
	"math/rand"
	"time"

	"github.com/po3rin/gonnp/entity"
	"github.com/po3rin/gonnp/matutils"
	"github.com/po3rin/gonnp/word"
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

	d, ok := x.(*mat.VecDense)
	if !ok {
		panic("gonnp: failed to transpose matrix to vec dense")
	}

	r, _ := d.Dims()
	dout := mat.NewDense(r, 1, d.RawVector().Data)
	dv := mat.NewVecDense(r, dout.RawMatrix().Data)
	x = matutils.MulMatVec(h, dv)

	_ = e.Embed.Backward(x)

	dh := matutils.MulMatVec(targetW, dv)
	return dh
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

type Sampler interface {
	GetNegativeSample(target mat.Vector) mat.Matrix
}

// UnigramSampler makes probability distribution of words from corpus
type UnigramSampler struct {
	SampleSize int
	VocabSize  int
	WordP      mat.Vector
}

// InitUnigraSampler inits UnigramSampler for Negative-Sampling.
func InitUnigraSampler(corpus word.Corpus, power float64, sampleSize int) *UnigramSampler {

	counts := make(map[int]float64, len(corpus))
	for _, id := range corpus {
		counts[int(id)]++
	}

	vocabSize := len(counts)
	wordP := mat.NewVecDense(vocabSize, nil)

	for i := 0; i < vocabSize; i++ {
		wordP.SetVec(i, counts[i])
	}

	w := mat.NewDense(vocabSize, 1, nil)

	pow := func(i, j int, v float64) float64 {
		return math.Pow(v, power)
	}
	w.Apply(pow, wordP)
	w.Scale(1/mat.Sum(w), w)

	s := &UnigramSampler{
		SampleSize: sampleSize,
		VocabSize:  vocabSize,
		WordP:      w.ColView(0),
	}

	return s
}

// GetNegativeSample gets negative sampling.
func (u *UnigramSampler) GetNegativeSample(target mat.Vector) mat.Matrix {
	batchSize, _ := target.Dims()
	negativeSample := mat.NewDense(batchSize, u.SampleSize, nil)

	p := u.WordP
	for i := 0; i < batchSize; i++ {
		v := mat.VecDenseCopyOf(p)
		targetIDx := target.AtVec(i)

		v.SetVec(int(targetIDx), 0)
		v.ScaleVec(1/mat.Sum(v), v)

		fs, err := weightedChoice(u.VocabSize, u.SampleSize, v.RawVector().Data)
		if err != nil {
			panic(err)
		}

		negativeSample.SetRow(i, fs)
	}
	return negativeSample
}

var randGenerator = func(max float64) float64 {
	rand.Seed(time.Now().UnixNano())
	r := rand.Float64() * max
	return r
}

// weightedChoice choice num wirh weight. Deduplication is default.
// ref: https://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
// ToDo: refacts deduplication & error.
func weightedChoice(v, size int, w []float64) ([]float64, error) {
	// convert v to slice.
	vs := make([]int, 0, v)
	for i := 0; i < v; i++ {
		vs = append(vs, i)
	}

	result := make([]float64, 0, size)
	for i := 0; i < size; i++ {
		var sum float64
		for _, v := range w {
			sum += v
		}

		r := randGenerator(sum)

		for j, v := range vs {
			r -= w[j]
			if r < 0 {
				result = append(result, float64(v))

				// delete choiced item.
				// https://github.com/golang/go/wiki/SliceTricks#delete
				w = append(w[:j], w[j+1:]...)
				vs = append(vs[:j], vs[j+1:]...)

				break
			}
		}
	}
	return result, nil
}

// NegativeSamplingLoss is layer for negative sampling.
type NegativeSamplingLoss struct {
	SampleSize     int
	EmbedDotLayers []*EmbeddingDot
	LossLayers     []LossLayer
	Sampler        Sampler
}

// // InitNegativeSamplingLoss inits NegativeSamplingLoss.
// func InitNegativeSamplingLoss(
// 	weight mat.Matrix,
// 	corpus word.Corpus,
// 	sampler Sampler,
// 	power float64,
// 	sampleSize int,
// ) *NegativeSamplingLoss {
// 	lossLayers := make([]LossLayer, 0, sampleSize+1)
// 	embedDotLayers := make([]*EmbeddingDot, 0, sampleSize+1)

// 	for i := 0; i < sampleSize+1; i++ {
// 		lossLayers = append(lossLayers, InitSigmoidWithLossLayer())
// 		embedDotLayers = append(embedDotLayers, InitEmbeddingDotLayer(weight))
// 	}

// 	return &NegativeSamplingLoss{
// 		SampleSize:     sampleSize,
// 		EmbedDotLayers: embedDotLayers,
// 		LossLayers:     lossLayers,
// 		Sampler:        sampler,
// 	}
// }

// // Forward calicurates loss with negative sampling.
// func (n *NegativeSamplingLoss) Forward(h, target mat.Matrix) float64 {
// 	batchSize, _ := target.Dims()
// 	td := mat.DenseCopyOf(target)
// 	vt := td.RowView(0)

// 	negativeSample := n.Sampler.GetNegativeSample(vt)
// 	nsd := mat.DenseCopyOf(negativeSample)

// 	// correct forward
// 	score := n.EmbedDotLayers[0].Forward(h, target)
// 	s := make([]float64, batchSize)
// 	for i := 0; i < batchSize; i++ {
// 		s[i] = 1
// 	}
// 	correctLabel := mat.NewDense(batchSize, 1, s)
// 	loss := n.LossLayers[0].Forward(score, correctLabel)

// 	// negative forward
// 	negativeLabel := mat.NewDense(batchSize, 1, nil)
// 	for i := 0; i < n.SampleSize; i++ {
// 		negativeTarget := nsd.ColView(i)
// 		score := n.EmbedDotLayers[1+i].Forward(h, negativeTarget)
// 		negativeLoss := n.LossLayers[1+i].Forward(score, negativeLabel)
// 		loss += negativeLoss
// 	}

// 	return loss
// }
