// +build !e2e

package nn_test

import (
	"testing"

	"github.com/po3rin/gonlp/nn"
)

func TestNewTwoLayer(t *testing.T) {
	tests := []struct {
		name       string
		inputSize  int
		hiddenSize int
		outputSize int
	}{
		{
			name:       "5:3:2",
			inputSize:  5,
			hiddenSize: 3,
			outputSize: 2,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			nn := nn.NewTwoLayerNet(tt.inputSize, tt.hiddenSize, tt.outputSize)
			if nn.NodeSize.Input != tt.inputSize ||
				nn.NodeSize.Hidden != tt.hiddenSize ||
				nn.NodeSize.Output != tt.outputSize {
				t.Errorf("unexpected node size")
			}

			// TODO: add assert.
		})
	}

}
