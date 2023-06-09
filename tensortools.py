import numpy as np

class Tensor():
    def __init__(self, data, label: str='', requires_grad: bool=False, _components: tuple=(), _operator: str='') -> None:
        self.data = np.array(data, dtype=float)
        self.label = label
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self.shape = self.data.shape
        self._components = set(_components)
        self._operator = _operator

    def __repr__(self):
        return f'Tensor({self.data})'

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, label=str(other))

        out = Tensor(self.data + other.data, requires_grad=True, _components=(self, other), _operator='+')

        def _backward():
            self.grad = out.grad
            other.grad = out.grad

        out._backward = _backward
        return out
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, label=str(other))

        out = Tensor(self.data - other.data, requires_grad=True, _components=(self, other), _operator='-')

        if self.requires_grad:
            def _backward():
                self.grad = -out.grad
                other.grad = -out.grad 
            out._backward = _backward

        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, label=str(other))
        
        out = Tensor(self.data * other.data, requires_grad=True, _components=(self, other), _operator='*')

        if self.requires_grad:
            def _backward():
                self.grad = -out.grad
                other.grad = -out.grad  
            out._backward = _backward

        return out

    def __pow__(self, other: int):
        out = Tensor(self.data**other, requires_grad=True, _components=(self, Tensor(other, label=str(other))), _operator='**')
        return out

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, label=str(other))
        
        out = Tensor(self.data * other.data**-1, requires_grad=True, _components=(self, other), _operator='/')

        if self.requires_grad:
            def _backward():
                self.grad = other.data**-1 * out.grad
                other.grad = -self.grad * other.data**-2 * out.grad
            out._backward = _backward

        return out
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, label=str(other))
        
        out = Tensor(self.data @ other.data, requires_grad=True, _components=(self, other), _operator='@')

        if self.requires_grad:
            def _backward():
                self.grad = out.grad @ other.data.T
                other.grad = self.data.T @ out.grad
            out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=True, _components=(self,), _operator='exp')

        if self.requires_grad:
            def _backward():
                self.grad = np.exp(self.data) * out.grad
            out._backward = _backward

        return out
    
    def __neg__(self):
        out = Tensor(-self.data, requires_grad=True, _components=(self,), _operator='neg')

        if self.requires_grad:
            def _backward():
                self.grad = -out.grad
            out._backward = _backward

        return out
    
    def T(self):
        out = Tensor(self.data.T, requires_grad=True, _components=(self,), _operator='T')

        if self.requires_grad:
            def _backward():
                self.grad = out.grad.T
            out._backward = _backward

        return out
    

def zeros(shape: tuple, label: str='', requires_grad=True):
    return Tensor(np.zeros(shape), label=label, requires_grad=requires_grad)

def randn(shape: tuple, label: str='', requires_grad=True):
    return Tensor(np.random.randn(*shape), label=label, requires_grad=requires_grad)


def draw_function(root: Tensor):
    # code by Andrej Karpathy https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1
    from graphviz import Digraph

    def trace(root):
        nodes, edges = set(), set()
        def build(v):

            if v not in nodes:
                nodes.add(v)

                for component in v._components:
                    edges.add((component, v))
                    build(component)

        build(root)
        return nodes, edges

    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = '{ %s | shape %s }' % (n.label, str(n.shape)), shape='record')

        if n._operator:
            dot.node(name = uid + n._operator, label = n._operator)
            dot.edge(uid + n._operator, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._operator)

    return dot