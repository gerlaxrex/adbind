import math
import random
import adbind as ab
import time

if __name__ == "__main__":
    # Simple example with a small function
    x = ab.Variable(2.0)
    y = ab.Variable(5.0)
    z = ab.sin(ab.log(x**x) + ab.exp(y))
    z.backward()

    print(z)
    print(x)
    print(y)

    # Now try to use backpropagation in order to change the weights of a linear function
    def function(x: float, y: float, w1: ab.Variable, w2: ab.Variable, b: ab.Variable):
        return b + w1*x + w2*y

    def real_process(x: float, y: float):
        return 1.0 + 2.5*x -0.32*y

    inputs = [(random.random(), random.random()) for _ in range(100)]
    
    w1 = ab.Variable(0.2)
    w2 = ab.Variable(0.3)
    b = ab.Variable(0.0)
    epochs = 10
    lr = 0.1

    # training with sgd
    st = time.perf_counter()
    for epoch in range(epochs):
        print(f"#### Epoch {epoch}")
        for idx, (x, y) in enumerate(inputs):
            y_hat = function(x, y, w1, w2, b)
            y_real = real_process(x, y)
            
            loss = (y_hat - y_real) ** 2
            
            loss.backward()
            if idx % 20 == 0:
                print(f"Loss variable: {loss}")
            
            # GD
            w1.set_value(w1.get_value() - lr * w1.get_grad())
            w2.set_value(w2.get_value() - lr * w2.get_grad())
            b.set_value(b.get_value() - lr * b.get_grad())
            # reset the grad to zero
            w1.reset()
            w2.reset()
            b.reset()
    
    print(f"Finished in {time.perf_counter() - st}")

    y_hats = [function(x,y,w1,w2,b).get_value() for x,y in inputs]
    y_reals = [real_process(x,y) for x,y in inputs]

    # print(y_hats)

    # print(y_reals)