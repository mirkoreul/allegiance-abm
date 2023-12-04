""" Functions
Helper functions that may be used in several scripts to perform simple tasks.
"""

# DEPENDENCIES
import numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def in_range(f, r):
    """
    Check if a float is within range of values.

    Parameters
    ----------
    f : float
        Single number float value that is to be tested.
    r : tuple
        Tuple with two elements indicating the lower and upper boundaries of the range, respectively.

    Returns
    -------
    boolean
        True if the number is within range.
    """
    return r[0] <= f <= r[1]


def pop_none(l):
    """
    Remove None elements from a list.

    Also deals with None values in lists that are nested one level.

    Parameters
    ----------
    l : List
        The list that is to be cleared of None values.

    Returns
    -------
    List
        List without None values.
    """
    if not any(isinstance(v, list) for v in l):
        return [v for v in l if v is not None]
    else:
        return [[v for v in nl if v is not None] for nl in l if nl is not None]


def isnested(l):
    """
    Check if list values are stored in a nested list.

    This is typically the case when the tracker is used to store attribute values of all agents for each iteration,
    rather than a single value for each iteration.

    Parameters
    ----------
    l : List
        List to check.

    Returns
    -------
    bool
        True if values are nested.
    """
    return any([isinstance(v, list) for v in l])


def nest_level(obj):
    """
    Get nesting level of list-type objects.

    Source: https://stackoverflow.com/questions/42727586/nest-level-of-a-list
    """
    # Not a list? So the nest level will always be 0:
    if type(obj) != list:
        return 0
    # Now we're dealing only with list objects:
    max_level = 0
    for item in obj:
        # Getting recursively the level for each item in the list,
        # then updating the max found level:
        max_level = max(max_level, nest_level(item))
    # Adding 1, because 'obj' is a list (here is the recursion magic):
    return max_level + 1


def softmax(vector):
    """Normalize vector to softmax probability distribution"""
    return numpy.exp(vector) / numpy.sum(numpy.exp(vector))


def random_exp_unit(lam=1):
    """
    Generate random number following truncated negative exponential distribution in the unit interval [0, 1].

    Source: https://stackoverflow.com/a/20408401

    Parameters
    ----------
    lam : float
        Rate parameter.

    Returns
    -------
    float
        Random number.
    """
    return -numpy.log(1 - (1 - numpy.exp(-lam)) * numpy.random.uniform()) / lam


def lowess(x, y, f=1. / 3.):
    """
    Basic LOWESS smoother with uncertainty.

    Proximity weighting of points using tricube kernel smoother function.

    Note:
        - Not robust (so no iteration) and only normally distributed errors.
        - No higher order polynomials d=1 so linear smoother.

    Source: https://james-brennan.github.io/posts/lowess_conf/

    Parameters
    ----------
    x : List<float>
        Observations x-coordinate.
    y : List<float>
        Observations y-coordinate.
    f : float
        Reduction factor

    Returns
    -------

    """
    # get some paras
    xwidth = f * (x.max() - x.min())  # effective width after reduction factor
    n = len(x)
    order = numpy.argsort(x)

    # storage
    y_sm = numpy.zeros_like(y)
    y_stderr = numpy.zeros_like(y)

    # weighting function
    def tricube(d):
        return numpy.clip((1 - numpy.abs(d) ** 3) ** 3, 0, 1)

    # run the regression for each observation i
    for i in range(n):
        dist = numpy.abs((x[order][i] - x[order])) / xwidth
        w = tricube(dist)

        # form linear system with the weights
        a = numpy.stack([w, x[order] * w]).T
        b = w * y[order]
        ata = a.T.dot(a)
        a_tb = a.T.dot(b)

        # solve the system
        sol = numpy.linalg.solve(ata, a_tb)

        # predict for the observation only
        yest = a[i].dot(sol)  # equiv of a.dot(yest) just for k
        place = order[i]
        y_sm[place] = yest
        sigma2 = (numpy.sum((a.dot(sol) - y[order]) ** 2) / n)

        # calculate the standard error
        y_stderr[place] = numpy.sqrt(sigma2 *
                                     a[i].dot(numpy.linalg.inv(ata)
                                              ).dot(a[i]))
    return y_sm, y_stderr


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=0.15, alpha=1.0,
              ax=None):
    """
    Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
    The color is taken from optional data in z, and creates a LineCollection.

    z can be:
    - empty, in which case a default coloring will be used based on the position along the input arrays
    - a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
    - an array of the length of at least the same length as x, to color according to this data
    - an array of a smaller length, in which case the colors are repeated along the curve

    The function colorline returns the LineCollection created, which can be modified afterwards.
    It plots a colored line with coordinates x and y, and optionally specified colors in the array z.

    Source: https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    See also: plt.streamplot
    See also: https://matplotlib.org/devdocs/gallery/lines_bars_and_markers/multicolored_line.html
    """
    def make_segments(xs, ys):
        """
        Create list of line segments from x and y coordinates, in the correct format for LineCollection:
        an array of the form numlines x (points per line) x 2 (x and y) array.
        """
        points = numpy.array([xs, ys]).T.reshape(-1, 1, 2)
        return numpy.concatenate([points[:-1], points[1:]], axis=1)

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = numpy.array([z])

    z = numpy.asarray(z)
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc
