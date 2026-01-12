from utils import Variant
from .tensor import Tensor
from ..ops.creation import full
from collections import Dict, List, Optional


struct MoTree(Copyable, ImplicitlyCopyable, Movable):
    comptime Treeable = Variant[Self, Tensor, Int, Float32, Dict[String, Self]]
    var data: List[Self.Treeable]

    fn __init__(out self) raises:
        self.data = List[Self.Treeable]()

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
    fn __init__(out self, var init_data: Dict[String, Self]) raises:
        self.data = List[Self.Treeable]()
        self.data.append(init_data^)

    @implicit
    fn __init__(out self, init_data: List[Self]) raises:
        self.data = List[Self.Treeable]()
        for d in init_data:
            self.data.append(d)

    @implicit
    fn __init__(out self, init_data: List[Tensor]) raises:
        self.data = List[Self.Treeable]()
        for d in init_data:
            self.data.append(MoTree(d))

    @implicit
    fn __init__(out self, *init_data: Self) raises:
        self.data = List[Self.Treeable]()
        for d in init_data:
            self.data.append(d)

    fn __copyinit__(out self, existing: Self):
        self.data = List[Self.Treeable](existing.data)

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data^

    fn __setitem__(mut self, _key: Variant[String, Int], value: Self.Treeable) raises:
        var mut_Val = value

        if _key.isa[Int]():
            var idx = _key[Int]
            if idx >= len(self.data):
                raise Error("Index out of range in setitem for MoTree")

            if mut_Val.isa[Self]():
                self.data[idx] = mut_Val.take[Self]()
            elif mut_Val.isa[Tensor]():
                self.data[idx] = MoTree(mut_Val.take[Tensor]())
            elif mut_Val.isa[Int]():
                self.data[idx] = MoTree(mut_Val.take[Int]())
            elif mut_Val.isa[Float32]():
                self.data[idx] = MoTree(mut_Val.take[Float32]())
            elif mut_Val.isa[Dict[String, Self]]():
                self.data[idx] = mut_Val.take[Dict[String, Self]]()
            else:
                raise Error("Cannot setitem of MoTree on unknown Type.")

        elif _key.isa[String]():
            if len(self.data) == 0:
                self.data.append(Dict[String, Self]())

            if not self.data[0].isa[Dict[String, Self]]():
                raise Error("MoTree is not a dictionary!")

            var key = _key[String]

            if mut_Val.isa[Self]():
                self.data[0][Dict[String, Self]][key] = mut_Val.take[Self]()
            elif mut_Val.isa[Tensor]():
                self.data[0][Dict[String, Self]][key] = MoTree(mut_Val.take[Tensor]())
            elif mut_Val.isa[Int]():
                self.data[0][Dict[String, Self]][key] = MoTree(mut_Val.take[Int]())
            elif mut_Val.isa[Float32]():
                self.data[0][Dict[String, Self]][key] = MoTree(mut_Val.take[Float32]())
            else:
                raise Error("Cannot setitem of MoTree on unknown Type.")

    fn __getitem__(ref self, key: Variant[String, Int]) raises -> Self:
        if key.isa[String]():
            var k = key[String]
            if len(self.data) == 1:
                if self.data[0].isa[Dict[String, Self]]():
                    return self.data[0][Dict[String, Self]][k]
                else:
                    raise Error("Unsupported return type in getitem")
            else:
                raise Error("Cannot get item with key: " + k)

        elif key.isa[Int]():
            var idx = key[Int]
            if idx < len(self.data):
                if self.data[idx].isa[Self]():
                    return self.data[idx][Self]
                else:
                    raise Error("Cannot index into a Leaf node")
            else:
                raise Error("Cannot get item at idx: " + String(idx))
        else:
            raise Error("Cannot retrieve value, unknown key")

    fn as_tensor(self) raises -> Tensor:
        if self.data[0].isa[Tensor]():
            return self.data[0][Tensor]
        elif self.data[0].isa[Int]():
            return full(self.data[0][Int], [], DType.int32)
        elif self.data[0].isa[Float32]():
            return full(self.data[0][Int], [], DType.float32)  # Fixed type
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
        _retrieve_tensors_rec(self, tensors)
        return tensors^

    fn flatten(self) raises -> Tuple[Self, List[Tensor]]:
        var leaves = self.get_all_tensors()
        var treedef = self  # Implicit copy
        return treedef, leaves^

    @staticmethod
    fn unflatten(treedef: Self, leaves: List[Tensor]) raises -> Self:
        var reconstructed = treedef  # Implicit copy
        var leaf_idx = 0
        _unflatten_rec(reconstructed, leaves, leaf_idx)
        return reconstructed^

    fn __getattr__(ref self, name: String) raises -> Self:
        return self[name]

    fn __setattr__(mut self, name: String, var val: Self) raises:
        self[name] = val^

    # Overloads for implicit conversion via setattr
    fn __setattr__(mut self, name: String, val: Tensor) raises:
        self[name] = MoTree(val)

    fn __setattr__(mut self, name: String, val: Int) raises:
        self[name] = MoTree(val)

    fn __setattr__(mut self, name: String, val: Float32) raises:
        self[name] = MoTree(val)

    fn __setattr__(mut self, name: String, var val: Dict[String, Self]) raises:
        self[name] = MoTree(val^)

    fn __setattr__(mut self, name: String, val: List[Tensor]) raises:
        self[name] = MoTree(val)


fn _retrieve_tensors_rec(curr: MoTree, mut tensors: List[Tensor]) raises -> None:
    for val in curr.data:
        if val.isa[Tensor]():
            tensors.append(val[Tensor])
        elif val.isa[MoTree]():
            _retrieve_tensors_rec(val[MoTree], tensors)
        elif val.isa[Dict[String, MoTree]]():
            for value in val[Dict[String, MoTree]].values():
                _retrieve_tensors_rec(value, tensors)
        else:
            pass


fn _unflatten_rec(mut curr: MoTree, leaves: List[Tensor], mut leaf_idx: Int) raises:
    for i in range(len(curr.data)):
        if curr.data[i].isa[Tensor]():
            curr.data[i] = leaves[leaf_idx]
            leaf_idx += 1
        elif curr.data[i].isa[MoTree]():
            var sub_tree = curr.data[i].take[MoTree]()
            _unflatten_rec(sub_tree, leaves, leaf_idx)
            curr.data[i] = sub_tree^
        elif curr.data[i].isa[Dict[String, MoTree]]():
            var dict = curr.data[i].take[Dict[String, MoTree]]()
            var keys = List[String]()
            for key_ref in dict.keys():
                keys.append(key_ref)  # key_ref is String

            for k in keys:
                var key = k  # k is already String or convertible
                if dict.__contains__(key):
                    var sub = dict[key]
                    _unflatten_rec(sub, leaves, leaf_idx)
                    dict[key] = sub^
            curr.data[i] = dict^
