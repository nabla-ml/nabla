import nabla_new as nb


fn main() raises:
    print("Testing Nabla Refactor...")

    # Test Tensor creation
    var x = nb.randn([2, 2])
    print("Tensor created:", x.shape().__str__())

    # Test Ops
    var y = x + x
    print("Addition result shape:", y.shape().__str__())

    # Test NN
    var model = nb.MLP([2, 4, 1])
    var input = nb.randn([1, 2])
    var output = model.execute([input])
    print("MLP output shape:", output[0].shape().__str__())

    print("Refactor verification successful!")
