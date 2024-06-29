from typing import Any, Callable, List, Literal, Optional, Union


class BinNode:
    def __init__(self, data: Any) -> None:
        self.parent: Optional["BinNode"] = None
        self.l_child: Optional["BinNode"] = None
        self.r_child: Optional["BinNode"] = None
        self.data = data

    def insert_lc(self, child: "BinNode"):
        self.l_child = child
        child.parent = self
        return self.l_child

    def insert_rc(self, child: "BinNode"):
        self.r_child = child
        child.parent = self
        return self.r_child

    def is_root(self) -> bool:
        if self.parent is None:
            return True
        else:
            return False

    def is_l_child(self) -> bool:
        if not self.is_root() and self.parent.l_child is self:
            return True
        else:
            return False

    def is_r_child(self) -> bool:
        if not self.is_root() and self.parent.r_child is self:
            return True
        else:
            return False

    def has_child(self) -> bool:
        if self.l_child is not None or self.r_child is not None:
            return True
        else:
            return False

    def has_sibling(self) -> bool:
        if self.is_l_child() and self.parent.r_child is not None:
            return True
        elif self.is_r_child() and self.parent.l_child is not None:
            return True
        else:
            return False


class BinArraySearchNode(BinNode):
    def __init__(
        self, data: Optional[list], status: Union[Literal["input"], Literal["result"]] = "input"
    ) -> None:
        super().__init__(data)
        self.status = status
        self.comp_func = mock_func

    def has_child(self) -> bool:
        if self.l_child is not None:
            return True
        else:
            return False

    def has_r_sibling(self) -> bool:
        if self.is_l_child() and self.parent.r_child is not None:
            return True
        else:
            return False

    def construct_subtree(self) -> Literal[0]:
        node = self
        data = self.data
        lo = 0
        hi = len(data)

        while 1 < hi - lo:
            mi = (lo + hi) // 2
            res_l = self.comp_func(data[lo:mi])
            res_r = self.comp_func(data[mi:hi])

            if isinstance(res_l, list) and isinstance(res_r, list):
                node.insert_lc(BinArraySearchNode(res_l, "result"))
                node.insert_rc(BinArraySearchNode(res_r, "result"))
                return 0

            elif res_l == -1 and res_r == -1:
                node.insert_rc(BinArraySearchNode(data[mi:hi]))
                node = node.insert_lc(BinArraySearchNode(data[lo:mi]))
                hi = mi

            elif res_l == -1 and isinstance(res_r, list):
                node.insert_rc(BinArraySearchNode(res_r, "result"))
                node = node.insert_lc(BinArraySearchNode(data[lo:mi]))
                hi = mi

            elif isinstance(res_l, list) and res_r == -1:
                node.insert_lc(BinArraySearchNode(res_l, "result"))
                node = node.insert_rc(BinArraySearchNode(data[mi:hi]))
                lo = mi

            else:
                assert False

        res_l = self.comp_func(data[lo:hi])
        res_l = [None] if res_l == -1 else res_l
        node.insert_lc(BinArraySearchNode(res_l, "result"))

        return 0

    def next_node(self) -> Optional["BinArraySearchNode"]:
        if self.has_child():
            return self.trav_left()
        elif self.has_r_sibling():
            return self.parent.r_child
        else:
            return None

    def trav_left(self) -> "BinArraySearchNode":
        node = self
        while node.has_child():
            node = node.l_child

        return node


class BinArraySearchTree:
    def __init__(self, array: list, comp_func: Callable) -> None:
        self.root = BinArraySearchNode(array)
        self.comp_func = comp_func
        self.root.construct_subtree()

    def get_results(self) -> list:
        self._results: List[Any] = []
        node = self.root
        self.trav_pre(node)
        return self._results

    def trav_left(self, node: BinArraySearchNode, stack: List[BinArraySearchNode]) -> None:
        while node.has_child():
            if node.r_child is not None:
                stack.append(node.r_child)

            node = node.l_child

        if node.status == "input":
            node.construct_subtree()
            self.trav_left(node, stack)
        elif node.status == "result":
            self._results.extend(node.data)
        else:
            assert False

    def trav_pre(self, node: BinArraySearchNode) -> None:
        stack: List[BinArraySearchNode] = []
        while True:
            self.trav_left(node, stack)
            if len(stack) == 0:
                break
            node = stack.pop()


def mock_func(array: List[int]) -> Union[List[float], Literal[-1]]:
    if 0 in array:
        return -1
    else:
        return [float(i) for i in array]


if __name__ == "__main__":
    mock_array0 = [0, 1, 2, 3, 4, 5]
    mock_array1 = [1, 2, 3, 4, 5, 0]
    mock_array2 = [0, 1, 2, 3, 4, 5, 0]
    mock_array3 = [0, 1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5, 0]
    mock_array5 = [0]
    mock_array6 = [1]
    search_tree = BinArraySearchTree(mock_array6, mock_func)
    print(search_tree.get_results())
