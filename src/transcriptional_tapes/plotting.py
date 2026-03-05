import matplotlib.pyplot as plt


def plot_single_trace(t, y):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, y)
    return fig, ax
