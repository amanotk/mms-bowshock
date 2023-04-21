# -*- coding: utf-8 -*-

"""Utilities

"""

import warnings

import copy
import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.interpolate as interpolate
import pandas as pd
import xarray as xr
import datashader
import PIL

try:
    import pytplot
except:
    pytplot = None

_default_layout = {
    'dpi'           : 300,
    'width'         : 800,
    'height'        : 800,
    'vspace'        : 25,
    'margin_top'    : 40,
    'margin_bottom' : 50,
    'margin_left'   : 100,
    'margin_right'  : 140,
    'line_width'    : 1,
    'fontsize'      : 12,
    'labelsize'     : 12,
    'ticklength'    : 6,
    'tickwidth'     : 1,
    'tickpad'       : 2,
    'colorbar_sep'  : 20,
    'colorbar_size' : 25,
}

_tplot_default_attrs = {
    'plot_options' : {
        'xaxis_opt' : {
            'axis_label' : 'Time',
            'crosshair' : 'X',
            'x_axis_type' : 'linear',
        },
        'yaxis_opt' : {
            'axis_label' : 'Y',
            'crosshair' : 'Y',
            'y_axis_type' : 'linear',
        },
        'zaxis_opt' : {
            'axis_label' : 'Z',
            'crosshair' : 'Z',
            'z_axis_type' : 'linear',
        },
        'extras' : {
            'spec' : False,
            'colormap' : ['viridis'],
            'panel_size' : 1,
            'char_size' : 10,
        },
        'trange' : [0.0, 1.0],
        'spec_bins_ascending' : None,
        'line_opt': {},
        'time_bar': [],
        'create_time': None,
        'links': {},
        'overplots': [],
        'interactive_xaxis_opt': {},
        'interactive_yaxis_opt': {},
        'colorbar_ticks' : None,
        'error' : None,
    },
}

_tplot_option_table = {
    # x axis
    'xlabel'        : ('xaxis_opt', 'axis_label', ),
    'x_label'       : ('xaxis_opt', 'axis_label', ),
    'xtype'         : ('xaxis_opt', 'x_axis_type', ),
    'x_type'        : ('xaxis_opt', 'x_axis_type', ),
    'xrange'        : ('xaxis_opt', 'x_range', ),
    'x_range'       : ('xaxis_opt', 'x_range', ),
    # y axis
    'ylabel'        : ('yaxis_opt', 'axis_label', ),
    'y_label'       : ('yaxis_opt', 'axis_label', ),
    'ytype'         : ('yaxis_opt', 'y_axis_type', ),
    'y_type'        : ('yaxis_opt', 'y_axis_type', ),
    'yrange'        : ('yaxis_opt', 'y_range', ),
    'y_range'       : ('yaxis_opt', 'y_range', ),
    'legend'        : ('yaxis_opt', 'legend_names', ),
    # z axis
    'zlabel'        : ('zaxis_opt', 'axis_label', ),
    'z_label'       : ('zaxis_opt', 'axis_label', ),
    'ztype'         : ('zaxis_opt', 'z_axis_type', ),
    'z_type'        : ('zaxis_opt', 'z_axis_type', ),
    'zrange'        : ('zaxis_opt', 'z_range', ),
    'z_range'       : ('zaxis_opt', 'z_range', ),
    # other
    'trange'        : ('trange', ),
    't_range'       : ('trange', ),
    'fontsize'      : ('extras', 'char_size', ),
    'char_size'     : ('extras', 'char_size', ),
    'linecolor'     : ('extras', 'line_color', ),
    'line_color'    : ('extras', 'line_color', ),
    'colormap'      : ('extras', 'colormap', ),
    'panelsize'     : ('extras', 'panel_size',),
    'cb_ticks'      : ('colorbar_ticks',),
    'colorbar_ticks': ('colorbar_ticks',),
    'ysubtitle'     : ('yaxis_opt', 'axis_subtitle',),
}


def get_default_layout():
    return copy.deepcopy(_default_layout)


def get_default_tplot_attrs():
    return copy.deepcopy(_tplot_default_attrs)


def get_tplot_option_table():
    return copy.deepcopy(_tplot_option_table)


def is_ipython():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False

def is_jupyter():
    import sys
    return 'ipykernel' in sys.modules


def cast_xarray(var):
    "cast input (scalar or sequence) into xarray's DataArray"
    if isinstance(var, str) and pytplot is not None:
        return pytplot.data_quants[var]
    elif isinstance(var, xr.DataArray):
        return var
    elif hasattr(var, '__iter__'):
        return list([cast_xarray(v) for v in var])
    else:
        raise ValueError('Unrecognized input')


def cast_list(var):
    if not isinstance(var, list):
        return list([var])
    else:
        return var


def process_kwargs(opt, kwargs, key, newkey=None):
    if newkey is None:
        newkey = key
    if key in kwargs:
        opt[key] = kwargs[key]


def set_plot_option(data, **kwargs):
    option_table = get_tplot_option_table()
    option_keys = option_table.keys()

    # check
    plot_options = data.attrs.get('plot_options', None)
    if plot_options is None:
        raise ValueError('Invalid input DataArray')

    # set options
    for key in kwargs.keys():
        if key in option_keys:
            try:
                table  = option_table[key]
                option = plot_options
                for i in range(len(table)-1):
                    option = option.get(table[i])
                option[table[-1]] = kwargs[key]
            except:
                raise warnings.warn('Error in setting option : %s' % (key))
        else:
            pass


def get_plot_option(data, key, val=None):
    option_table = get_tplot_option_table()
    option_keys = option_table.keys()

    # check
    plot_options = data.attrs.get('plot_options', None)
    if plot_options is None:
        raise ValueError('Invalid input DataArray')

    # set options
    if key in option_keys:
        try:
            table  = option_table[key]
            option = plot_options
            for i in range(len(table)-1):
                option = option.get(table[i])
            return option[table[-1]]
        except:
            pass

    return val


def get_figure_class(var, classdict):
    opt = var.attrs['plot_options'].get('extras')

    if opt.get('plotter', None) is not None:
        return opt.get('plotter')

    if opt.get('spec', False) and opt.get('spec'):
        return classdict.get('Spec')

    if opt.get('alt', False) and opt.get('alt'):
        return classdict.get('Alt')

    if opt.get('map', False) and opt.get('map'):
        return classdict.get('Map')

    return classdict.get('Line')


def get_figure_layout(var, **kwargs):
    var = cast_list(cast_xarray(var))

    layout = get_default_layout()
    for key in layout.keys():
        if key in kwargs:
            layout[key] = kwargs[key]

    # work in unit of pixels
    fig_h    = layout['height']
    fig_w    = layout['width']
    margin_t = layout['margin_top']
    margin_b = layout['margin_bottom']
    margin_l = layout['margin_left']
    margin_r = layout['margin_right']
    vspace   = layout['vspace']

    # var_label
    if 'var_label' in kwargs:
        print('Warning: var_label functionality has not yet been implemented')

    # get unit size for each panel in pixels
    N  = len(var)
    ps = [0] * N
    for i in range(N):
        if isinstance(var[i], xr.DataArray):
            ps[i] = get_plot_option(var[i], 'panelsize')
        elif hasattr(var[i], '__iter__'):
            ps[i] = get_plot_option(var[i][0], 'panelsize')
    ps = np.array(ps)
    ph = (fig_h - (margin_t + margin_b + vspace*(N-1))) / N
    pw = (fig_w - (margin_l + margin_r))
    hh = ph * ps
    ww = np.ones((N,)) * pw
    vv = np.ones((N,)) * vspace

    # bounding box in pixels
    x0 = np.zeros_like(hh)
    x1 = np.zeros_like(hh)
    y0 = np.zeros_like(hh)
    y1 = np.zeros_like(hh)
    x0[ : ] = margin_l
    x1[ : ] = x0[:] + ww
    y0[  0] = margin_b
    y0[1:N] = y0[0] + np.cumsum(hh + vv)[0:N-1]
    y1[ : ] = y0[:] + hh

    # reverse order
    x0 = x0[::-1]
    x1 = x1[::-1]
    y0 = y0[::-1]
    y1 = y1[::-1]

    bbox_pixels = {
        'x0' : x0,
        'x1' : x1,
        'y0' : y0,
        'y1' : y1,
    }

    bbox_relative = {
        'x0' : x0 / fig_w,
        'x1' : x1 / fig_w,
        'y0' : y0 / fig_h,
        'y1' : y1 / fig_h,
    }

    layout['bbox_pixels']   = bbox_pixels
    layout['bbox_relative'] = bbox_relative
    return layout


def bbox_to_rect(bbox):
    l = bbox['x0']
    b = bbox['y0']
    w = bbox['x1'] - bbox['x0']
    h = bbox['y1'] - bbox['y0']
    return l, b, w, h


def interpolate_spectrogram(ybin, data, **kwargs):
    from scipy import interpolate
    def interp(x, y, newx):
        f = interpolate.interp1d(x, y, axis=0, kind='nearest',
                                 bounds_error=False, fill_value=None)
        return f(newx)

    nx = data.shape[0]

    # check ybin
    if ybin.ndim == 1:
        y0 = ybin[ 0]
        y1 = ybin[-1]
        ny = ybin.shape[-1]
        yy = np.tile(ybin, (nx,1))
    elif ybin.ndim == 2:
        y0 = ybin[:, 0].min()
        y1 = ybin[:,-1].max()
        ny = ybin.shape[-1]
        yy = ybin
    else:
        raise ValueError('Invalid input')

    # bin centers and edges
    my = 2*ny
    if 'ylog' in kwargs and kwargs['ylog']:
        binc = np.logspace(np.log10(y0), np.log10(y1), my)
    else:
        binc = np.linspace(y0, y1, my)
    bine = np.concatenate([binc[:+1], 0.5*(binc[+1:] + binc[:-1]), binc[-1:]])

    zz = np.zeros((nx, my), np.float64)
    for ii in range(nx):
        zz[ii,:] = interp(yy[ii,:], data[ii,:], binc)

    opt = dict(y0=y0, y1=y1, bine=bine, binc=binc)

    return zz, opt


def prepare_raster_spectrogram(x, y, z, **kwargs):
    ylog = kwargs.get('ylog', False)
    zlog = kwargs.get('zlog', False)

    # asjust array shape
    zz = z.T
    Ny, Nx = zz.shape

    x = to_unixtime(x)
    if x.ndim == 1:
        xx = np.tile(x[np.newaxis,:], (Ny, 1))
    elif x.ndim == 2:
        xx = x.T
    else:
        raise ValueError('invalid shape of x')

    if y.ndim == 1:
        yy = np.tile(y[:,np.newaxis], (1, Nx))
    elif y.ndim == 2:
        yy = y.T
    else:
        raise ValueError('invalid shape of y')

    # preprocess
    if ylog:
        yy   = np.log10(yy)

    if zlog:
        zz   = np.log10(zz)
        zz   = np.where(np.isinf(zz), np.nan, zz)

    # data for rasterization
    data = xr.DataArray(zz, name='z',
                        dims=['ydim', 'xdim'],
                        coords={'y' : (['ydim', 'xdim'], yy),
                                'x' : (['ydim', 'xdim'], xx)})

    xcoord = np.sort(np.unique(xx))
    ycoord = np.sort(np.unique(yy))
    xmin = xcoord[ 0] - 0.5*(xcoord[ 1] - xcoord[ 0])
    xmax = xcoord[-1] + 0.5*(xcoord[-1] - xcoord[-2])
    ymin = ycoord[ 0] - 0.5*(ycoord[ 1] - ycoord[ 0])
    ymax = ycoord[-1] + 0.5*(ycoord[-1] - ycoord[-2])
    zmin = np.nanmin(zz)
    zmax = np.nanmax(zz)

    # attach attributes for rasterization
    data.attrs['ylog'] = ylog
    data.attrs['zlog'] = zlog
    data.attrs['xmin'] = xmin
    data.attrs['xmax'] = xmax
    data.attrs['ymin'] = ymin
    data.attrs['ymax'] = ymax
    data.attrs['zmin'] = zmin
    data.attrs['zmax'] = zmax

    return data


def do_raster_spectrogram(data, **kwargs):
    from matplotlib import cm
    shade = datashader.transfer_functions.shade

    # high resolution rasterization for zooming
    zoom_ratio = 4
    width   = kwargs.get('width')  * zoom_ratio
    height  = kwargs.get('height') * zoom_ratio
    x_range = kwargs.get('x_range', [data.attrs['xmin'], data.attrs['xmax']])
    y_range = kwargs.get('y_range', [data.attrs['ymin'], data.attrs['ymax']])
    z_range = kwargs.get('z_range', [data.attrs['zmin'], data.attrs['zmax']])
    cmap = kwargs.get('cmap', 'viridis')

    canvas_opts = {
        'x_range' : x_range,
        'y_range' : y_range,
        'plot_width' : int(width),
        'plot_height' : int(height),
    }
    canvas = datashader.Canvas(**canvas_opts)

    shade_opts = {
        'how' : 'linear',
        'cmap' : cm.get_cmap(cmap),
        'span' : z_range,
    }
    image = shade(canvas.quadmesh(data, x='x', y='y'), **shade_opts).to_pil()

    return image


def get_ds_raster_spectrogram(x, y, z, **kwargs):
    from matplotlib import cm
    shade = datashader.transfer_functions.shade

    cmap = kwargs.get('cmap', 'viridis')
    ylog = kwargs.get('ylog', False)
    zlog = kwargs.get('zlog', False)
    zmin = kwargs.get('zmin', None)
    zmax = kwargs.get('zmax', None)
    width = kwargs.get('width', None)
    height = kwargs.get('height', None)

    # asjust array shape
    zz = z.T
    Ny, Nx = zz.shape

    if x.ndim == 1:
        xx = np.tile(x[np.newaxis,:], (Ny, 1))
    elif x.ndim == 2:
        xx = x.T
    else:
        raise ValueError('invalid shape of x')

    if y.ndim == 1:
        yy = np.tile(y[:,np.newaxis], (1, Nx))
    elif y.ndim == 2:
        yy = y.T
    else:
        raise ValueError('invalid shape of y')

    # preprocess
    if ylog:
        yy   = np.log10(yy)

    if zlog:
        zz   = np.log10(zz)
        zz   = np.where(np.isinf(zz), np.nan, zz)
        zmin = np.nanmin(zz) if zmin is None else np.log10(zmin)
        zmax = np.nanmax(zz) if zmax is None else np.log10(zmax)
    else:
        zmin = np.nanmin(zz) if zmin is None else zmin
        zmax = np.nanmax(zz) if zmax is None else zmax

    xcoord = np.sort(np.unique(xx))
    ycoord = np.sort(np.unique(yy))
    xmin = xcoord[ 0] - 0.5*(xcoord[ 1] - xcoord[ 0])
    xmax = xcoord[-1] + 0.5*(xcoord[-1] - xcoord[-2])
    ymin = ycoord[ 0] - 0.5*(ycoord[ 1] - ycoord[ 0])
    ymax = ycoord[-1] + 0.5*(ycoord[-1] - ycoord[-2])

    # rasterization via datashader
    data = xr.DataArray(zz, name='z',
                        dims=['ydim', 'xdim'],
                        coords={'y' : (['ydim', 'xdim'], yy),
                                'x' : (['ydim', 'xdim'], xx)})
    canvas_opts = {
        'x_range' : (xmin, xmax),
        'y_range' : (ymin, ymax),
        'plot_width' : int(width),
        'plot_height' : int(height),
    }
    canvas = datashader.Canvas(**canvas_opts)

    shade_opts = {
        'how' : 'linear',
        'cmap' : cm.get_cmap(cmap),
        'span' : [zmin, zmax],
    }
    image = shade(canvas.quadmesh(data, x='x', y='y'), **shade_opts).to_pil()

    # return other parameters as a dict
    if ylog:
        ymin = 10**ymin
        ymax = 10**ymax

    if zlog:
        zmin = 10**zmin
        zmax = 10**zmax

    opt = {
        'xmin' : xmin,
        'xmax' : xmax,
        'ymin' : ymin,
        'ymax' : ymax,
        'zmin' : zmin,
        'zmax' : zmax,
    }

    return image, opt


def get_raster_spectrogram(y, z, Ny=None, ylog=False, zlog=False,
                           zmin=None, zmax=None, cmap=None):
    from matplotlib import cm, colors

    def interp(x, y, newx):
        f = interpolate.interp1d(x, y, axis=0, kind='nearest',
                                 bounds_error=False, fill_value=None)
        return f(newx)

    # default colormap
    if cmap is None:
        cmap = 'viridis'

    # keep structure in the first dimension (time)
    Nx = z.shape[0]

    # twice the original
    if Ny is None:
        Ny = z.shape[1] * 2

    # y can be 2D for time-varying bins
    if y.ndim == 1:
        y0 = y[ 0]
        y1 = y[-1]
        yy = np.tile(y, (Nx,1))
    elif y.ndim == 2:
        y0 = y[ 0,:].min()
        y1 = y[-1,:].max()
        yy = y
    else:
        raise ValueError('Error: invalid input')

    # set new bin
    if ylog:
        ybin = np.logspace(np.log10(y0), np.log10(y1), Ny)
    else:
        ybin = np.linspace(y0, y1, Ny)

    # interpolation in y
    zz = np.zeros((Nx, Ny), np.float64)
    for ii in range(Nx):
        zz[ii,:] = interp(yy[ii,:], z[ii,:], ybin)

    # preprocess
    if zlog:
        zind = np.logical_and(np.isfinite(zz), np.greater(zz, 0.0))
        zmin = zz[zind].min() if zmin is None else zmin
        zmax = zz[zind].max() if zmax is None else zmax
        norm = colors.LogNorm(vmin=zmin, vmax=zmax)
    else:
        zind = np.isfinite(zz)
        zmin = zz[zind].min() if zmin is None else zmin
        zmax = zz[zind].max() if zmax is None else zmax
        norm = colors.Normalize(vmin=zmin, vmax=zmax)

    # get rasterized image
    zmask = ma.masked_array(zz, mask=~zind)
    colormap = cm.get_cmap(cmap)
    rgbarray = np.uint8(colormap(norm(zmask[:,::-1].T))*255)
    pilimage = PIL.Image.fromarray(rgbarray)

    # return other parameters as a dict
    opt = dict(y0=y0, y1=y1, ybin=ybin, zmin=zmin, zmax=zmax, norm=norm)

    return pilimage, opt

def get_raster_colorbar(N=None, cmap=None):
    from matplotlib import cm

    # default size of lut
    if N is None:
        N = 256

    # default colormap
    if cmap is None:
        cmap = 'viridis'

    # get rasterized image of N x 1
    z = np.linspace(0.0, 1.0, N)[::-1,None]
    colormap = cm.get_cmap(cmap)
    rgbarray = np.uint8(colormap(z)*255)
    pilimage = PIL.Image.fromarray(rgbarray)

    return pilimage


def time_slice(var, t1, t2):
    t1 = pd.Timestamp(t1).timestamp()
    t2 = pd.Timestamp(t2).timestamp()

    var = cast_list(cast_xarray(var))
    ret = [v.loc[t1:t2] for v in var]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def pd_to_datetime(t):
    tt = np.atleast_1d(np.array(t))
    dt = tt.dtype
    if isinstance(tt[0], str):
        tt = pd.to_datetime(tt)
    elif dt == np.float32 or dt == np.float64:
        tt = pd.to_datetime(tt, unit='s')
    elif dt == object and isinstance(tt[0], pd.Timestamp):
        tt = pd.DatetimeIndex(tt.astype(np.datetime64))
    elif dt == np.dtype('datetime64[ns]'):
        tt = pd.to_datetime(tt, unit='s')
    elif dt == object:
        tt = pd.to_datetime(tt)
    else:
        raise ValueError('Unrecognized time format : ', dt)
    return tt


def to_scalar_or_array(t):
    if np.isscalar(t):
        return t
    elif t.size > 1:
        return t
    else:
        return t[0]


def to_unixtime(t):
    tt = pd_to_datetime(t).values.astype(np.int64) * 1.0e-9
    return to_scalar_or_array(tt)


def to_datetime64(t):
    tt = pd_to_datetime(t).values
    return to_scalar_or_array(tt)


def to_pydatetime(t):
    tt = pd_to_datetime(t).to_pydatetime()
    return to_scalar_or_array(tt)


def to_datestring(t, fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    tt = pd_to_datetime(t).strftime(fmt)
    return to_scalar_or_array(tt)


def create_xarray(**data):
    if 'x' in data and 'y' in data:
        x = np.array(data['x'])
        y = np.array(data['y'])
        # check compatibility
        if x.ndim == 1 and y.ndim == 1 and x.size == y.size:
            dims = ('time',)
            v = None
        elif x.ndim == 1 and y.ndim == 2 and x.size == y.shape[0]:
            dims = ('time', 'vdim')
            v = np.arange(y.shape[1])
        else:
            raise ValueError('Error: incompatible input')
    else:
            raise ValueError('Error: incompatible input')

    if 'name' in data:
        name = data['name']
    else:
        name = None

    # create DataArray object
    if v is not None:
        obj = xr.DataArray(y, dims=dims, coords={'time' : ('time', x), 'v' : ('vdim', v)})
    else:
        obj = xr.DataArray(y, dims=dims, coords={'time' : ('time', x)})
    obj.name = name
    obj.attrs = get_default_tplot_attrs()

    return obj


def sph2xyz(r, t, p, degree=True):
    if degree:
        t = np.deg2rad(t)
        p = np.deg2rad(p)
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return x, y, z


def xyz2sph(x, y, z, degree=True):
    r = np.sqrt(x**2 + y**2 + z**2)
    t = np.arccos(z / r)
    p = np.arctan2(y, x)
    if degree:
        t = np.rad2deg(t)
        p = np.rad2deg(p)
    return r, t, p
