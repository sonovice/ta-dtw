import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")
GOLDEN_RATIO = (math.sqrt(5) - 1) / 2


def save_fig_decorator(**kwargs):
    plot_width = kwargs.get('plot_width') or 1.0
    font_size = kwargs.get('font_size') or 10
    ratio = kwargs.get('ratio') or GOLDEN_RATIO

    text_width_pt = 412.56503
    text_width_inches = text_width_pt / 72.27

    def decorator(func):
        def wrap(*args, **kwargs):
            mpl.use('pgf')
            params = {
                # 'text.usetex': True,
                'font.serif': ['Minion Math'],
                'font.sans-serif': [
                    'Myriad Pro',
                    'Inconsolata',
                ],
                'font.family': 'sans-serif',

                'font.size': font_size,
                'xtick.labelsize': font_size,
                'ytick.labelsize': font_size,
                'axes.labelsize': font_size,
                'axes.titlesize': font_size,
                'legend.title_fontsize': font_size,
                'legend.fontsize': font_size,

                'figure.constrained_layout.use': True,
                'figure.figsize': (
                    text_width_inches * plot_width,
                    text_width_inches * plot_width * ratio
                )
            }
            mpl.rcParams.update(params)

            # actual plot call
            func(*args, **kwargs)

            basename = func.__name__
            plt.savefig(f'{basename}.pdf')

        return wrap
    return decorator
