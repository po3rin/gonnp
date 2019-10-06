// Package optimizers updates params (ex. weight, bias ...) using various algorism.
package optimizers

import "github.com/po3rin/gonnp/params"

// Optimizer updates prams.
type Optimizer interface {
	Update(params []params.Param, grads []params.Grad) []params.Param
}
