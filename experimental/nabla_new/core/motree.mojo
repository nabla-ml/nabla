from utils import Variant
from .tensor import Tensor
from ..ops.creation import full


struct MoTree(Copyable, Movable):
    alias Treeable = Variant[Self, Tensor, Int, Float32, Dict[String, Self]]
    var data: List[Self.Treeable]

    fn __init__(out self) raises:
        self.data = []

    @implicit
    fn __init__(out self, init_data: Tensor) raises:
        self.data = List[Self.Treeable]()
        self.data.append(init_data)

    @implicit
    fn __init__(out self, init_data: Int) raises:
        self.data = List[Self.Treeable]()
        self.data.append(init_data)

    @implicit
    fn __init__(out self, init_data: Float32) raises:
        self.data = List[Self.Treeable]()
        self.data.append(init_data)

    @implicit
    fn __init__(out self, init_data: Dict[String, Self]) raises:
        self.data = List[Self.Treeable]()
        self.data.append(init_data.copy())

    @implicit
    fn __init__(out self, init_data: List[Self]) raises:
        self.data = List[Self.Treeable]()
        for d in init_data:
            self.data.append(d.copy())

        if not self.data[0].isa[Tensor]():
            raise "Error: The list shoudl contain of Tensors directly"

    @implicit
    fn __init__(out self, init_data: List[Tensor]) raises:
        self.data = List[Self.Treeable]()
        for d in init_data:
            self.data.append(d.copy())

    @implicit
    fn __init__(out self, *init_data: Self) raises:
        self.data = List[Self.Treeable]()
        for d in init_data:
            self.data.append(d.copy())

    fn __setitem__(mut self, _key: Variant[String, Int], value: Self.Treeable) raises:
        if _key.isa[Int]():
            var idx = _key[Int]
            if idx >= len(self.data):
                raise Error("Index out of range in seetitem for MoTree")

            if value.isa[Self]():
                self.data[idx] = value[Self].copy()
            elif value.isa[Tensor]():
                self.data[idx] = value[Tensor]
            elif value.isa[Int]():
                self.data[idx] = value[Int]
            elif value.isa[Float32]():
                self.data[idx] = value[Float32]
            elif value.isa[Dict[String, Self]]():
                self.data[idx] = value[Dict[String, Self]].copy()
            else:
                raise Error("Cannot setitem of MoTree on unknown Type.")

        elif _key.isa[String]():
            if len(self.data) == 0:
                self.data.append(Dict[String, Self]())

            if not self.data[0].isa[Dict[String, Self]]():
                raise Error("MoTree is not a dictionary!")

            var key = _key[String]
            if value.isa[Self]():
                self.data[0][Dict[String, Self]][key] = value[Self].copy()
            elif value.isa[Tensor]():
                self.data[0][Dict[String, Self]][key] = value[Tensor]
            elif value.isa[Int]():
                self.data[0][Dict[String, Self]][key] = value[Int]
            elif value.isa[Float32]():
                self.data[0][Dict[String, Self]][key] = value[Float32]
            else:
                raise Error("Cannot setitem of MoTree on unknown Type.")

    fn __getitem__(self, key: Variant[String, Int]) raises -> Self:
        if key.isa[String]():
            var k = key[String]
            if len(self.data) == 1:
                if self.data[0].isa[Dict[String, Self]]():
                    return self.data[0][Dict[String, Self]][k].copy()
                else:
                    raise Error("Unknsupported return type in getitem")
            else:
                raise Error("Cannot get item with key:", k)

        elif key.isa[Int]():
            var idx = key[Int]
            if idx < len(self.data):
                if self.data[idx].isa[Self]():
                    return self.data[idx][Self].copy()
                elif self.data[idx].isa[Tensor]():
                    return self.data[idx][Tensor]
                elif self.data[idx].isa[Int]():
                    return self.data[idx][Int]
                elif self.data[idx].isa[Float32]():
                    return self.data[idx][Float32]
                elif self.data[idx].isa[Dict[String, Self]]():
                    return self.data[idx][Dict[String, Self]]
                else:
                    raise Error("Unknsupported return type in getitem")
            else:
                raise Error("Cannot get item at idx:", idx)
        else:
            raise Error("Cannot retreive value, unknown key")

    fn as_tensor(self) raises -> Tensor:
        if self.data[0].isa[Tensor]():
            return self.data[0][Tensor]
        elif self.data[0].isa[Int]():
            return full(self.data[0][Int], [], DType.int32)
        elif self.data[0].isa[Float32]():
            return full(self.data[0][Int], [], DType.int32)
        else:
            raise Error("Value cannot be converted to a Tensor")

    fn as_int(self) raises -> Int:
        if self.data[0].isa[Int]():
            return self.data[0][Int]
        elif self.data[0].isa[Float32]():
            return Int(self.data[0][Float32])
        else:
            raise Error("Value cannot be converted to an Int")

    fn as_float32(self) raises -> Float32:
        if self.data[0].isa[Int]():
            return Float32(self.data[0][Int])
        elif self.data[0].isa[Float32]():
            return self.data[0][Float32]
        else:
            raise Error("Value cannot be converted to a Float32")

    fn get_all_tensors(self) raises -> List[Tensor]:
        var tensors = List[Tensor]()
        _retreive_tensors_rec(self, tensors)
        return tensors^

    fn __getattr__(self, name: String) raises -> Self:
        return self[name]

    fn __setattr__(mut self, name: String, val: Self) raises:
        self[name] = val.copy()


fn _retreive_tensors_rec(curr: MoTree, mut tensors: List[Tensor]) raises -> None:
    for val in curr.data:
        if val.isa[Tensor]():
            tensors.append(val[Tensor])
        elif val.isa[Dict[String, MoTree]]():
            for value in val[Dict[String, MoTree]].values():
                _retreive_tensors_rec(value, tensors)
        else:
            return
