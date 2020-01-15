from matplotlib import pyplot as plt
import numpy as np
from fqi.utils import action_cols, state_cols
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KDTree
import matplotlib

def plot_track(track_in, track_out, x=[], y=[]):

    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.plot(track_in.x, track_in.y, 'k', linewidth=0.5)
    ax.plot(track_out.x, track_out.y, 'k', linewidth=0.5)

    if len(x) > 0:
        fx = x[0]
        fy = y[0]
        ex = x[-1]
        ey = y[-1]

        # find closest in boundary point
        kdt_in = KDTree(track_in[['x', 'y']].values)
        d, i = kdt_in.query(np.array([[fx, fy],[ex, ey]]), k=1)
        start_in_x = track_in.x.values[i[0]]
        start_in_y = track_in.y.values[i[0]]
        end_in_x = track_in.x.values[i[1]]
        end_in_y = track_in.y.values[i[1]]

        # find closest out boundary point
        kdt_out = KDTree(track_out[['x', 'y']].values)
        d, i = kdt_out.query(np.array([[fx, fy],[ex, ey]]))
        start_out_x = track_out.x.values[i[0]]
        start_out_y = track_out.y.values[i[0]]
        end_out_x = track_out.x.values[i[1]]
        end_out_y = track_out.y.values[i[1]]

        ax.plot([start_in_x, start_out_x], [start_in_y, start_out_y], 'g')
        ax.plot([end_in_x, end_out_x], [end_in_y, end_out_y], 'r')

    return f, ax

def plot_trajectories(simulation, lap, f, ax, ref=[], save_fig=False, path='./'):

    lap_mask = simulation['NLap'] == lap
    if len(ref) == 0:
        p_ref = ax.plot(simulation['xCarWorld'][simulation['isReference']],
                simulation['yCarWorld'][simulation['isReference']], 'r')
    else:
        p_ref = ax.plot(ref['xCarWorld'], ref['yCarWorld'], 'r')

    p_lap = ax.plot(simulation['xCarWorld'][lap_mask], simulation['yCarWorld'][lap_mask])

    lap_time = simulation['time'][lap_mask].values[-1]

    if len(ref) == 0:
        ref_time = simulation['time'][simulation['isReference']].values[-1]
    else:
        ref_time = ref['time'].values[-1]

    ax.set_title('Lap {} - {} s / {} s'.format(lap, lap_time, ref_time))

    ax.legend((p_ref[0], p_lap[0]), ('Reference', 'Lap'))

    if save_fig:
        f.savefig(path + '{}_trajectory.svg'.format(lap), format='svg')

def plot_q_delta(lap, simulation, evaluation, f, ax, save_fig=False, path='./'):

    lap_mask = simulation['NLap'] == lap

    policyQ = evaluation[lap][1]
    pilotQ = evaluation[lap][0]

    value = policyQ - pilotQ
    bound = max((abs(min(value)), abs(max(value))))
    s = ax.scatter(simulation['xCarWorld'][lap_mask], simulation['yCarWorld'][lap_mask],
                   s=2, c=value, vmin=0, vmax=bound)
    f.colorbar(s, ax=ax)
    ax.set_title('Lap ' + str(lap) + ' Policy Q - Pilot Q')

    if save_fig:
        f.savefig(path + '{}_q_delta.svg'.format(lap), format='svg')

def plot_q(lap, simulation, evaluation, f, ax, q_name, save_fig=False, path='./'):

    lap_mask = simulation['NLap'] == lap

    if q_name == 'policy':
        value = evaluation[lap][1]
    else:
        value = evaluation[lap][0]

    s = ax.scatter(simulation['xCarWorld'][lap_mask], simulation['yCarWorld'][lap_mask],
                   s=2, c=value)
    f.colorbar(s, ax=ax)
    ax.set_title('Lap ' + str(lap) + ' Pilot Q')

    if save_fig:
        f.savefig(path + '{}_{}_q.svg'.format(lap, q_name), format='svg')


def plot_action_delta(lap, action_pos, simulation, evaluation, f, ax, save_fig=False, path='./'):

    lap_mask = simulation['NLap'] == lap

    policyA = evaluation[lap][2][:, action_pos]
    pilotA = np.array(simulation[lap_mask][action_cols[action_pos]])

    value = policyA - pilotA

    minv = min(value)
    maxv = max(value)
    bound = max((abs(minv), abs(maxv)))

    """s = ax.scatter(simulation['xCarWorld'][lap_mask], simulation['yCarWorld'][lap_mask],
                   s=5, c=value, cmap='seismic', vmin=-bound, vmax=bound)
    'Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'])"""
    s = ax.scatter(simulation['xCarWorld'][lap_mask], simulation['yCarWorld'][lap_mask],
                   s=5, c=value, cmap='autumn', vmin=-bound, vmax=bound)

    if action_cols[action_pos] == 'aSteerWheel':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.5)
        cbar = f.colorbar(s, ax=ax, cax=cax, orientation='horizontal')
        cbar.ax.invert_xaxis()
    else:
        f.colorbar(s, ax=ax)

    ax.set_title('Lap ' + str(lap) + ' ' + action_cols[action_pos] + ' Policy - Pilot')
    if save_fig:
        f.savefig(path + '{}_trajectory_{}.svg'.format(lap, action_cols[action_pos]), format='svg')

    f = plt.figure()
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    if action_cols[action_pos] == 'aSteerWheel':
        ax.plot(value, range(len(value)))
        ax.invert_xaxis()

    else:
        ax.plot(value)
    ax.set_title('Lap ' + str(lap) + ' ' + action_cols[action_pos] + ' Policy - Pilot')
    if save_fig:
        f.savefig(path + '{}_diff_{}.svg'.format(lap, action_cols[action_pos]), format='svg')

    f = plt.figure()
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    if action_cols[action_pos] == 'aSteerWheel':
        value = np.array(simulation[action_cols[action_pos]][lap_mask])
        ax.plot(value, range(len(value)))
        value = evaluation[lap][2][:, action_pos]
        ax.plot(value, range(len(value)))
        ax.invert_xaxis()
    else:
        ax.plot(np.array(simulation[action_cols[action_pos]][lap_mask]))
        ax.plot(evaluation[lap][2][:, action_pos])

    ax.legend(('pilot', 'policy'))
    ax.set_title('Lap ' + str(lap) + ' ' + action_cols[action_pos] + ' Action values')
    if save_fig:
        f.savefig(path + '{}_{}.svg'.format(lap, action_cols[action_pos]), format='svg')

def plot_computation_times(maxq_time, fit_time, save_fig=False, path='./'):

    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    max_bp = ax.boxplot(maxq_time, patch_artist=True)
    ax.set_ylabel('Max Q time', color='green')
    [patch.set(facecolor='lightgreen') for patch in max_bp['boxes']]

    ax2 = ax.twinx()
    fit_bp = ax2.boxplot(fit_time, patch_artist=True)
    ax2.set_ylabel('Fit time', color='blue')
    [patch.set(facecolor='lightblue', alpha=0.7) for patch in fit_bp['boxes']]

    if save_fig:
        f.savefig(path + 'computation_times.svg', format='svg')

def plot_feature_importance(variables, importance, save_fig=False, path='./'):

    f = plt.figure(figsize=[20,20])
    ax = f.add_axes([0.45, 0.1, 0.55, 0.8])

    imp_idx = np.argsort(importance)

    n_variables = len(importance)

    sorted_variables = [variables[i] for i in imp_idx]

    ax.barh(range(n_variables), importance[imp_idx])
    c = ['red' if i in action_cols else 'black' for i in sorted_variables]
    ticks = plt.yticks(range(n_variables), sorted_variables)
    ax.set_title('Feature importance of Extratrees of Q', size=20)
    [ax.get_yticklabels()[i].set_color(c[i]) for i in range(n_variables)]
    ax.tick_params(labelsize=20)

    if save_fig:
        f.savefig(path + 'feature_importance.svg', format='svg')

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
