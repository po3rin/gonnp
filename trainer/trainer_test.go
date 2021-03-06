// +build !e2e

package trainer_test

import (
	"testing"

	"github.com/po3rin/gomnist"
	"github.com/po3rin/gonnp/models"
	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/trainer"
)

func TestFit(t *testing.T) {
	model := models.NewTwoLayerNet(784, 50, 10)
	optimizer := optimizers.InitSDG(0.01)
	trainer := trainer.InitTrainer(model, optimizer, trainer.EvalInterval(10))

	l := gomnist.NewLoader("./../testdata", gomnist.OneHotLabel(true))
	mnist, err := l.Load()
	if err != nil {
		t.Fatalf("unexpected error: %v\n", err)
	}

	_ = trainer
	_ = mnist

	// checks no panic ...
	trainer.Fit(mnist.TestData, mnist.TestLabels, 1, 1)
}
