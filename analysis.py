#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
import json

import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
import xarray as xr

import pytplot
from aspy import set_plot_option, create_xarray

JSON_FILENAME = "shockgeometry.json"
WAVE_FILENAME = "burstwave_{:s}.h5"
TSERIES_FILENAME = "tseries_{:s}_mms{:1d}.png"
SCATTER_FILENAME = "scatter_{:s}_mms{:1d}.png"
DIR_FMT = "%Y%m%d_%H%M%S"
FCE_FMT = "mms{:1d}_fce"
FGM_FMT = "mms{:1d}_fgm_b_gse_srvy_l2"
PSD_FMT = "mms{:1d}_scm_psd"
BABS_FMT = "mms{:1d}_babs"
POWER_FMT = "mms{:1d}_power"

# minimum frequency for integration
fmin = [0.05, 0.10, 0.20]


def read_json(dirname):
    with open(os.sep.join([dirname, JSON_FILENAME]), "r") as fp:
        json_data = json.load(fp)
        json_data["ID"] = dirname
        json_data["B0"] = json_data["Bt1"][0]
        # check inbound/outbound
        t1 = pd.to_datetime(json_data["trange1"])
        t2 = pd.to_datetime(json_data["trange2"])
        json_data["is_inbound"] = t1[0] < t2[0]

    return json_data


def read_directory(json_data, sc, suffix):
    import download

    ID = json_data["ID"]

    # load data
    pytplot.del_data()
    fn_fast = os.sep.join([ID, "fast.h5"])
    fn_wave = os.sep.join([ID, WAVE_FILENAME.format(suffix)])
    download.load_hdf5(fn_fast, tplot=True)
    download.load_hdf5(fn_wave, tplot=True)

    # take average and interpolation to time coordinate of psd
    psd = pytplot.data_quants[PSD_FMT.format(sc)]
    fgm = pytplot.data_quants[FGM_FMT.format(sc)]
    fgm = fgm.rolling(time=16, center=True).mean().interp(time=psd.time)
    babs = create_xarray(name="babs_psd", x=fgm.time.values, y=fgm.values[:, 3])
    set_plot_option(babs, ylabel="|B| / B$_0$", line_color=("k",))

    # store data
    pytplot.data_quants[BABS_FMT.format(sc)] = babs


def get_transition_layer_mask(x, inbound, **kwargs):
    threshold = kwargs["threshold"]
    distance = kwargs["distance"]
    height = kwargs["height"]
    # select maximum
    peaks, _ = signal.find_peaks(x, distance=distance, height=height)
    iups = 0 if inbound else -1
    imax = peaks[x[peaks] >= x[peaks].max() * 0.80][iups]
    # apply threshold
    bmax = x[imax]
    bmin = (bmax - 1) * threshold + 1
    return ((inbound == False) ^ (np.arange(x.size) <= imax)) & (x >= bmin)


def gather_transition_layer(json_data, **kwargs):
    # take care of options
    sc = kwargs.get("sc", 1)
    suffix = kwargs.get("suffix", "2048")
    threshold = kwargs.get("threshold", 0.2)

    # read data
    read_directory(json_data, sc, suffix)

    inbound = json_data["is_inbound"]
    trange = pd.to_datetime(json_data["trange"])

    B0 = json_data["B0"]
    babs = pytplot.data_quants[BABS_FMT.format(sc)]
    time = babs.time
    power = pytplot.data_quants[POWER_FMT.format(sc)]
    valid = (
        np.isfinite(babs.values)
        & np.all(np.isfinite(power.values), axis=1)
        & np.all(power.values > 1e-12, axis=1)
    )
    index1 = np.argwhere(np.logical_and(valid, time.values > trange[0]))[0, 0]
    index2 = np.argwhere(np.logical_and(valid, time.values < trange[1]))[-1, 0]

    kwargs = {
        "threshold": threshold,
        "distance": 10,
        "height": 2,
    }
    t = time.values[index1:index2]
    x = babs.values[index1:index2] / B0
    y = power.values[index1:index2, :] / B0**2
    m = get_transition_layer_mask(x, inbound, **kwargs) & valid[index1:index2]
    t = t[m]
    x = x[m]
    y = y[m]

    if 0:
        if inbound:
            # inbound crossing
            n = index2 - index1
            t = time.values[index1:index2]
            x = babs.values[index1:index2] / B0
            y = power.values[index1:index2, :] / B0**2
            m = get_transition_layer_mask(x, threshold, inbound) & valid[index1:index2]
            t = t[m]
            x = x[m]
            y = y[m]
        else:
            # outbound crossing
            n = index2 - index1
            t = time.values[index1:index2]
            x = babs.values[index1:index2] / B0
            y = power.values[index1:index2, :] / B0**2
            m = get_transition_layer_mask(x, threshold, inbound) & valid[index1:index2]
            t = t[m]
            x = x[m]
            y = y[m]

    return t, x, y


def get_val_err(x):
    xx = np.quantile(x, [0.25, 0.75, 0.50], axis=0)
    val = xx[2, :]
    err = xx[0:2, :]
    err[0, :] = np.abs(err[0, :] - val)
    err[1, :] = np.abs(err[1, :] - val)
    return val, err


def plot_scatter(json_data, sc, suffix, x, y):
    import matplotlib as mpl
    from matplotlib import pylab as plt
    from matplotlib.gridspec import GridSpec

    ID = json_data["ID"]
    fontsize = 12

    fig = plt.figure(figsize=(10, 12))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.05, hspace=0.30, wspace=0.35)
    gs = GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

    # title
    title = "MMS{:1d} Bow Shock {:s}".format(sc, ID)
    fig.suptitle(title, fontsize=fontsize)
    fig.suptitle(title, fontsize=fontsize)

    # errorbar
    yval, yerr = get_val_err(y)

    for i in range(3):
        ax1 = plt.subplot(gs[i, 0])
        ax2 = plt.subplot(gs[i, 1])

        # scatter plot
        plt.sca(ax1)
        plt.scatter(x, y[:, i], s=10, marker="x", lw=1)
        plt.xlabel(r"|B| / B$_0$")
        plt.ylabel(r"Power (f$_{{\rm min}}$/f$_{{\rm ce}}$ = {:5.2f})".format(fmin[i]))

        # hisotgram
        plt.sca(ax2)
        nbin = 10
        bins = np.geomspace(y[:, i].min(), y[:, i].max(), nbin + 1)
        count, _ = np.histogram(y[:, i], bins=bins, density=False)
        plt.stairs(count, bins, orientation="horizontal", fill=True)
        # plot first to third quartiles
        plt.errorbar(
            [count.max() * 1.1],
            yval[i : i + 1],
            yerr=yerr[0, i : i + 1],
            fmt="o",
            ms=5,
            capsize=5,
            color="k",
        )
        plt.xlabel(r"Count")
        plt.ylabel(r"Power (f$_{{\rm min}}$/f$_{{\rm ce}}$ = {:4.2f})".format(fmin[i]))

        # scale
        for ax in (ax1, ax2):
            ax.grid()
            ax.set_yscale("log")
            ax.set_ylim(1e-7, 1e-1)

    # save file and close figure
    fig.savefig(os.sep.join([ID, SCATTER_FILENAME.format(suffix, sc)]))
    plt.close(fig)


def plot_timeseries(json_data, sc, suffix, t, x):
    import matplotlib as mpl
    from matplotlib import pylab as plt
    import pytplot
    from aspy import set_plot_option

    ID = json_data["ID"]
    B0 = json_data["B0"]
    t1 = pd.to_datetime(json_data["trange"][0]) - np.timedelta64(10, "s")
    t2 = pd.to_datetime(json_data["trange"][1]) + np.timedelta64(10, "s")
    trange = [t1, t2]
    fontsize = 12

    # store data to plot
    tplot_vars = [
        BABS_FMT.format(sc),
        POWER_FMT.format(sc),
        PSD_FMT.format(sc),
    ]
    for name in tplot_vars:
        set_plot_option(pytplot.data_quants[name], fontsize=fontsize)

    # figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(left=0.05, right=0.80, top=0.95, bottom=0.10, hspace=0.15)

    # title
    title = "MMS{:1d} Bow Shock {:s}".format(sc, ID)
    fig.suptitle(title, fontsize=fontsize)

    # suppress UserWarning in agg backend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytplot.tlimit([t.strftime("%Y-%m-%d %H:%M:%S") for t in trange])
        pytplot.tplot_options("axis_font_size", fontsize)
        pytplot.tplot(tplot_vars, fig=fig, axis=axs)

    # shock transition layer
    plt.sca(axs[0])
    plt.plot(t, x * B0, "kx")

    # power
    plt.sca(axs[1])
    # plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=fontsize)
    plt.legend(loc="upper right", fontsize=10)

    # spectrogram
    plt.sca(axs[2])
    fce = pytplot.data_quants[FCE_FMT.format(sc)]
    plt.plot(fce.time, 0.1 * fce.values, "w-", lw=1)
    plt.plot(fce.time, 0.5 * fce.values, "w-", lw=1)
    plt.plot(fce.time, 1.0 * fce.values, "w-", lw=1)

    # appearance
    if trange[1] - trange[0] > np.timedelta64(600, "s"):
        major_locator = mpl.dates.MinuteLocator(byminute=range(0, 60, 5))
        minor_locator = mpl.dates.SecondLocator(bysecond=range(0, 60, 10))
    elif trange[1] - trange[0] > np.timedelta64(180, "s"):
        major_locator = mpl.dates.MinuteLocator(byminute=range(0, 60, 1))
        minor_locator = mpl.dates.SecondLocator(bysecond=range(0, 60, 10))
    else:
        major_locator = mpl.dates.SecondLocator(bysecond=range(0, 60, 10))
        minor_locator = mpl.dates.SecondLocator(bysecond=range(0, 60, 1))

    for ax in axs:
        plt.sca(ax)
        plt.grid(True, linestyle="--")
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)

    axs[1].set_ylim(1e-6, 1e2)
    axs[1].yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=5))
    axs[-1].set_xlabel("UT", fontsize=fontsize)

    # save file and close figure
    fig.savefig(os.sep.join([ID, TSERIES_FILENAME.format(suffix, sc)]))
    plt.close(fig)


def doit(dirname, suffix, threshold):
    json_data = read_json(dirname)

    for sc in [1, 2, 3, 4]:
        print("{:s} : MMS{:1d}".format(dirname, sc))
        t, x, y = gather_transition_layer(json_data, sc=sc, threshold=threshold, suffix=suffix)
        plot_scatter(json_data, sc, suffix, x, y)
        plot_timeseries(json_data, sc, suffix, t, x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bust Wave Analysis Tool for MMS")
    parser.add_argument("target", nargs="+", type=str, help="target file or directory")
    parser.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=0.05,
        help="ratio of magnetic field strength relative to the maximum (overshoot)",
    )
    parser.add_argument(
        "--suffix",
        dest="suffix",
        type=str,
        default="2048",
        help="suffix for output filename",
    )
    args = parser.parse_args()

    threshold = args.threshold
    suffix = args.suffix

    ###
    ### analyze for each target
    ###
    import download

    for target in args.target:
        if os.path.isfile(target):
            #
            # event list file in CSV format
            #
            tr1, tr2 = download.read_eventlist(target)
            csv = pd.read_csv(target, header=None, skiprows=1)
            tr1 = pd.to_datetime(csv.iloc[:, 0])
            tr2 = pd.to_datetime(csv.iloc[:, 1])
            for (t1, t2) in zip(tr1, tr2):
                try:
                    dirname = t1.strftime(DIR_FMT) + "-" + t2.strftime(DIR_FMT)
                    doit(dirname, suffix, threshold)
                except Exception as e:
                    print(e)
                    print("Error: perhaps unrecognized directory format?")

        elif os.path.isdir(target):
            #
            # event directory
            #
            dirname = os.path.dirname(target + os.sep)
            tr = dirname.split("-")
            if len(tr) != 2:
                print("Error: unrecognized directory format")
                continue

            try:
                doit(dirname, suffix, threshold)
            except Exception as e:
                print(e)
                print("Error: perhaps unrecognized directory format?")

        else:
            print("Error: {} is not a file or directory".format(target))
