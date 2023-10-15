from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    if arg < 0 or arg >= len(vals):
        raise ValueError("Invalid 'arg' value")

    val_list = list(vals)

    x = val_list[arg] 
    x_plus_eps = x + epsilon
    x_minus_eps = x - epsilon

    val_list[arg] = x_plus_eps
    f_plus_eps = f(*val_list)

    val_list[arg] = x_minus_eps
    f_minus_eps = f(*val_list)

    central_diff = (f_plus_eps - f_minus_eps) / (2 * epsilon)

    return central_diff


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = set()
    res = []

    def visit(v:Variable) -> None:
        #pass if visited before
        if v.unique_id in visited:
            return
        if not v.is_leaf():
            for i in v.parents:
                if not i.is_constant():
                    visit(i)
        visited.add(v.unique_id)
        res.insert(0,v) 

    def visit_2(v:Variable) -> None:
        visited.add(v.unique_id)
        for i in v.parents:
            if not i.is_constant() and i.unique_id not in visited:
                visit(i)
        res.insert(0,v) 

    visit_2(variable)
    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    compute_graph = topological_sort(variable)
    deriv_dict = {}# node_id : derivative
    deriv_dict[variable.unique_id] = deriv

    for node in compute_graph:
        if node.is_leaf():
            continue
        node_deriv = deriv_dict[node.unique_id]
        inputs_grads = node.chain_rule(node_deriv)
        for input,grad in inputs_grads:
            if input.is_leaf():
                input.accumulate_derivative(grad)
                continue
            if input.unique_id in deriv_dict.keys():
                deriv_dict[input.unique_id] += grad
            else:
                deriv_dict[input.unique_id] = grad

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
