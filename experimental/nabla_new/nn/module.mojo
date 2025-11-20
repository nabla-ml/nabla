from nabla_new.core.tensor import Tensor
from nabla_new.core.execution import ExecutionContext, realize


trait Module(Copyable, Movable):
    fn __call__(self, args: List[Tensor]) raises -> List[Tensor]:
        ...

    fn params(self) raises -> List[Tensor]:
        ...

    fn ctx(self) raises -> ExecutionContext:
        ...

    fn execute(self, args: List[Tensor]) raises -> List[Tensor]:
        var all_args_have_data = True
        for arg in args:
            if not arg.has_data():
                all_args_have_data = False

        var outputs = self.__call__(args)

        if all_args_have_data:
            realize(outputs, self.ctx())

        return outputs^
