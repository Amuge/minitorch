from minitorch.module import Module, Parameter

class Module1(Module):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(5)
        self.a = Module2()
        self.b = Module3()


class Module2(Module):
    def __init__(self):
        super().__init__()
        self.p2 = Parameter(10)


class Module3(Module):
    def __init__(self):
        super().__init__()
        self.c = Module4()


class Module4(Module):
    def __init__(self):
        super().__init__()
        self.p3 = Parameter(15)

if __name__ == "__main__":
    print(Module1().named_parameters())
    print(Module1().modules())