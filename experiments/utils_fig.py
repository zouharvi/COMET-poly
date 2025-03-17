COLORS = [
    "#bc272d",  # red
    "#50ad9f",  # green
    "#0000a2",  # blue
    "#e9c716",  # yellow
    "#8c6e96",  # purple
]
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["legend.edgecolor"] = "None"
mpl.rcParams["legend.fontsize"] = 9
mpl.rcParams["legend.borderpad"] = 0.1


def turn_off_spines(which=['top', 'right'], ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    ax.spines[which].set_visible(False)
