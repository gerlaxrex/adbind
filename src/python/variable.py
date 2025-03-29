import math
import random

class Variable:
    def __init__(self, value, deps=None):
        self.value = value
        self.grad = None
        self.__deps = deps or []  
    
    def reset(self):
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value)
        out.__deps = [(self, 1.0), (other, 1.0)]
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value * other.value)
        out.__deps = [(self, other.value), (other, self.value)]
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value - other.value)
        out.__deps = [(self, 1.0), (other, -1.0)]
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        if other.value == 0:
            raise ZeroDivisionError("Division by zero in autodiff")
        out = Variable(self.value / other.value)
        out.__deps = [(self, 1.0 / other.value), (other, -self.value / (other.value ** 2))]
        return out

    def __pow__(self, other):
        if isinstance(other, Variable):
            if self.value <= 0:
                raise ValueError("Base must be positive for autodiff exponentiation")
            out = Variable(self.value ** other.value)
            out.__deps = [
                (self, other.value * (self.value ** (other.value - 1))),
                (other, (self.value ** other.value) * math.log(self.value))
            ]
        else:  
            out = Variable(self.value ** other)
            out.__deps = [(self, other * (self.value ** (other - 1)))]
        return out

    def __neg__(self):
        out = Variable(-self.value)
        out.__deps = [(self, -1.0)]
        return out

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rtruediv__ = __truediv__
    __rpow__ = __pow__

    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad})"

    def backward(self, adjoint=1.0):
        if self.grad is None:
            self.grad = 0.0
        self.grad += adjoint
        for node, w in self.__deps:
            node.backward(adjoint * w)


def log(x):
    if x.value <= 0:
        raise ValueError("Logarithm of non-positive value is undefined")
    v = Variable(math.log(x.value), deps=[(x, 1.0 / x.value)])
    return v

def exp(x):
    v = Variable(math.exp(x.value), deps=[(x, math.exp(x.value))])
    return v

def sin(x):
  return Variable(math.sin(x.value), deps=[(x, math.cos(x.value))])

def cos(x):
  return Variable(math.cos(x.value), deps=[(x, -math.sin(x.value))])

def relu(x):
    return Variable(max(0, x), deps=[(x, 1.0 if x > 0 else 0.0)])


if __name__ == "__main__":
    # Simple example with a small function
    x = Variable(2.0)
    y = Variable(5.0)
    z = sin(log(x**x) + exp(y))
    z.backward()

    print(z)
    print(x)
    print(y)

    # Now try to use backpropagation in order to change the weights of a linear function
    def function(x: float, y: float, w1: Variable, w2: Variable, b: Variable):
        return b + w1*x + w2*y

    def real_process(x: float, y: float):
        return 1.0 + 2.5*x -0.32*y

    inputs = [(random.random(), random.random()) for _ in range(100)]
    
    w1 = Variable(0.2)
    w2 = Variable(0.3)
    b = Variable(0.0)
    epochs = 2
    lr = 0.1

    # training with sgd
    for epoch in range(epochs):
        print(f"#### Epoch {epoch}")
        for idx, (x, y) in enumerate(inputs):
            y_hat = function(x, y, w1, w2, b)
            y_real = real_process(x, y)
            
            loss = (y_hat - y_real) ** 2
            
            loss.backward()
            if idx % 20 == 0:
                print(f"Loss variable: {loss}")
            
            # bp
            w1.value = w1.value - lr * w1.grad
            w2.value = w2.value - lr * w2.grad
            b.value = b.value - lr * b.grad
            # reset the grad to zero
            w1.reset()
            w2.reset()
            b.reset()

    y_hats = [function(x,y,w1,w2,b).value for x,y in inputs]
    y_reals = [real_process(x,y) for x,y in inputs]

    print(y_hats)

    print(y_reals)



