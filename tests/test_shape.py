import nabla 


def test_shape():
    shape0 = (2, 3)
    shape1 = (2, 2, 3)
    res_shape = nabla.get_broadcasted_shape(shape0, shape1)
    print(res_shape)


if __name__ == "__main__":
    test_shape()