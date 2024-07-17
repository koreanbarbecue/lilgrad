import math

class Value:

    def __init__(self, value, _children=(), _op='', label=''):
        self.value = value
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.value})"
    
    def __add__(self, other):
        out = Value(self.value + other.value, (self, other), '+')
        
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        out = Value(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad = other.value * out.grad
            other.grad = self.value * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward

        return out
    

if __name__ == "__main__":
    v = Value(10)
    c = Value(20)
    a = Value(5)
    d = v * c + a
    print(d._op)