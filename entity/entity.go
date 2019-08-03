// Package entity has common parametors type.
package entity

import (
	"gonum.org/v1/gonum/mat"
)

// Grad is gradient of wight & bias.
type Grad struct {
	Weight mat.Matrix
	Bias   mat.Vector
}

// Param has weight & bias.
type Param struct {
	Weight mat.Matrix
	Bias   mat.Vector
}

// ParamsManager manages params.
type ParamsManager interface {
	GetParams() []Param
	GetGrads() []Grad
	UpdateParams([]Param)
}

// ParamManager manages param.
type ParamManager interface {
	GetParam() Param
	GetGrad() Grad
	SetParam(p Param)
}
