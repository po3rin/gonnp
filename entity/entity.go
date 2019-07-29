package entity

import (
	"gonum.org/v1/gonum/mat"
)

// Grad is gradient of wight & bias.
type Grad struct {
	Weight mat.Matrix
	Bias   mat.Matrix
}

// Param has weight & bias.
type Param struct {
	Weight mat.Matrix
	Bias   mat.Matrix
}

// ParamsManager manages params.
type ParamsManager interface {
	GetParams() []Param
	GetGrads() []Grad
	SetParams(params []Param)
}

// ParamManager manages param.
type ParamManager interface {
	GetParam() Param
	GetGrad() Grad
}
