// Package params has common parametors type.
package params

import (
	"gonum.org/v1/gonum/mat"
)

// Param has weight & bias.
type Param struct {
	Weight  mat.Matrix
	WeightH mat.Matrix
	Bias    mat.Vector
}

// Grad is gradient of wight & bias.
type Grad struct {
	Weight  mat.Matrix
	WeightH mat.Matrix
	Bias    mat.Vector
}

// SetManager manages params.
type SetManager interface {
	GetParams() []Param
	GetGrads() []Grad
	UpdateParams([]Param)
}

// Manager manages param.
type Manager interface {
	GetParam() Param
	GetGrad() Grad
	SetParam(p Param)
}
