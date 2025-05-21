from python import Python, PythonObject
import nabla
import math
from time import perf_counter
from memory import memcpy


fn ground_truth(x: List[nabla.Array]) raises -> List[nabla.Array]:
    var res = nabla.sin(x[0] * Float32(2 * math.pi))
    return [
        res,
    ]


fn predictions(x: List[nabla.Array]) raises -> List[nabla.Array]:
    var res = nabla.sin(x[0] * Float32(2 * math.pi))
    var noise = nabla.randn((100,), mean=0, variance=0.1)
    res = res + noise
    return [
        res,
    ]


def test_animation():
    var animation = Python.import_module("matplotlib.animation")
    var FuncAnimation = animation.FuncAnimation
    var plt = Python.import_module("matplotlib.pyplot")

    var ctx = nabla.ExecutionContext()

    ground_truth_jitted = nabla.jit(ground_truth)
    predictions_jitted = nabla.jit(predictions)

    plots = plt.subplots()
    fig = plots[0]
    ax = plots[1]

    x_data_noisy = nabla.rand((100,))
    noise = nabla.randn((100,), mean=0, variance=0.1)
    y_data_noisy = predictions_jitted([x_data_noisy])[0]
    nabla.realize(y_data_noisy, ctx)

    plot_noisy = ax.plot(
        nabla.to_numpy(x_data_noisy),
        nabla.to_numpy(y_data_noisy),
        "o",
        alpha=0.6,
        label="Noisy Data",
    )
    line_noisy = plot_noisy[0]

    x_ground_truth = nabla.arange((500,)) / Float32(499)
    y_ground_truth = ground_truth_jitted([x_ground_truth])[0]
    nabla.realize(y_ground_truth, ctx)

    plot_ground_truth = ax.plot(
        nabla.to_numpy(x_ground_truth),
        nabla.to_numpy(y_ground_truth),
        "r-",
        label="Ground Truth Sine",
    )
    line_ground_truth = plot_ground_truth[0]

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.legend()

    plt.ion()

    # Animation loop
    for _ in range(200):
        x_data_noisy = nabla.rand((100,))
        noise = nabla.randn((100,), mean=0, variance=0.1)
        y_data_noisy = predictions_jitted([x_data_noisy])[0]
        nabla.realize(y_data_noisy, ctx)

        line_noisy.set_data(
            nabla.to_numpy(x_data_noisy), nabla.to_numpy(y_data_noisy)
        )  # Update the data for the noisy plot
        fig.canvas.draw_idle()
        plt.pause(0.05)

    # Keep the plot window open after animation completes
    plt.ioff()
    plt.show()
