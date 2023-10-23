#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl

mpl.use("Agg") if __name__ == "__main__" else None
from matplotlib import pyplot as plt

import pytplot
from utils import set_plot_option, create_xarray

DIR_FMT = "%Y%m%d_%H%M%S"
FGM_FMT = "mms{:1d}_fgm_b_gse_brst_l2"
SCM_FMT = "mms{:1d}_scm_acb_gse_scb_brst_l2"
EDP_FMT = "mms{:1d}_edp_dce_gse_brst_l2"
HDF_FILENAME = "burstwave_{:s}.h5"
PNG_FILENAME = "burstwave_{:s}_mms{:1d}.png"


class WaveAnalyzer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, trange, data_dict, dirname):
        import wavetools

        # prepare for output file
        filename = os.sep.join([dirname, HDF_FILENAME]).format(self.kwargs["suffix"])
        if os.path.exists(filename) and os.path.isfile(filename):
            os.remove(filename)

        zmin = self.kwargs.get("psd_min", 1e-7)
        zmax = self.kwargs.get("psd_max", 1e1)
        fgm = data_dict["fgm"]
        scm = data_dict["scm"]
        edp = data_dict["edp"]
        result = [None] * 4
        for i in range(4):
            if fgm[i] is None or scm[i] is None or edp[i] is None:
                continue
            result[i] = wavetools.msvd(edp[i], scm[i], fgm[i], **self.kwargs)
            result[i]["args"] = self.kwargs
            result[i]["scm"] = scm[i]
            result[i]["edp"] = edp[i]
            result[i]["fgm"] = fgm[i]
            result[i]["fgm"].name = "fgm"

            # psd
            scm_psd = result[i].pop("psd")
            set_plot_option(scm_psd, zrange=[zmin, zmax])
            result[i]["scm_psd"] = scm_psd

            # calculate fce
            fce = self.get_fce(scm_psd.time.values, fgm[i])
            result[i]["fce"] = fce

            # integrate psd in frquency
            fmin = [0.05, 0.10, 0.20]
            fmax = [1.00, 1.00, 1.00]
            power = self.get_integrated_power(scm_psd, fce, fmin, fmax)
            result[i]["power"] = power

            # summary plot
            t1 = pd.to_datetime(trange[0])
            t2 = pd.to_datetime(trange[1])
            plot_summary([t1, t2], i + 1, result[i], dirname, self.kwargs["suffix"])

            # save parameters
            result = save_parameters(i + 1, result[i], filename)

        return result

    def get_fce(self, tindex, fgm):
        # 1-sec moving averaged |B|
        sps = int(self.kwargs["sps_dcb"])
        babs = fgm[:, 3].rolling(time=sps).mean()

        # interpolate to index
        x = tindex
        y = 2.799e1 * babs.interp(time=x, method="linear").values

        # create DataArray object
        da = create_xarray(name="fce", x=x, y=y)
        set_plot_option(da, ylabel="fce [Freq]", line_color=("k",))
        return da

    def get_integrated_power(self, psd, fce, fmin, fmax):
        from scipy import interpolate
        from scipy import integrate

        ntime = psd.shape[0]
        psd_freq = psd.spec_bins.values.copy()
        psd_delf = psd_freq[1] - psd_freq[0]

        fmin = np.atleast_1d(fmin)
        fmax = np.atleast_1d(fmax)
        # check consistency
        if fmin.size != fmax.size:
            raise ValueError("invalid integration range!")

        nfrange = fmin.size

        # make sure that the time coordinates for psd and fce match
        if not np.alltrue(psd.time == fce.time):
            raise ValueError("coordinate mismatch detected!")

        index = np.arange(ntime, dtype=np.int32)
        power = np.zeros((ntime, nfrange), dtype=np.float64)

        mask = np.isnan(fce)
        index = np.ma.masked_array(index, mask=mask).compressed()

        for jj in range(nfrange):
            for ii in index:
                v = fce.values[ii]
                index_min = int(v * fmin[jj] / psd_delf) - 1
                index_max = int(v * fmax[jj] / psd_delf)
                # take care of bounding error
                index_min = np.clip(index_min, 0, psd.shape[1] - 1)
                index_max = np.clip(index_max, 0, psd.shape[1] - 1)
                x = psd_freq[index_min : index_max + 1]
                y = psd.values[ii, index_min : index_max + 1]
                # interpolate for integration
                f = interpolate.interp1d(x, y, bounds_error=False, fill_value="extrapolate")
                ii_min = 0
                ii_max = index_max - index_min
                xnew = x.copy()
                xnew[ii_min] = v * fmin[jj]
                xnew[ii_max] = v * fmax[jj]
                ynew = f(xnew)
                # store integrated power
                power[ii, jj] = integrate.trapezoid(ynew, x=xnew)

        # create DataArray object
        da = create_xarray(name="power", x=fce.time.values, y=power)
        legend = [r"f$_{{\rm min}}$/f$_{{\rm ce}}$ = {:4.2f}".format(f) for f in fmin]
        set_plot_option(
            da,
            ylabel=r"Integrated Power [nT$^2$]",
            ytype="log",
            yrange=[1e-6, 1e-0],
            line_color=("r", "g", "b"),
            legend=legend,
        )
        return da


def plot_summary(trange, sc, result, dirname, suffix):
    import matplotlib as mpl
    from matplotlib import pylab as plt
    import pytplot

    fontsize = 16

    # store data to plot
    pytplot.del_data()
    tplot_vars = [
        "fgm",
        "power",
        "scm_psd",
        "degpol",
        "planarity",
        "ellipticity",
        "theta_kb",
        "theta_sb",
    ]
    for name in tplot_vars:
        pytplot.data_quants[name] = result[name]
        set_plot_option(pytplot.data_quants[name], fontsize=fontsize)

    set_plot_option(
        pytplot.data_quants["fgm"],
        ylabel="B [nT]",
        ysubtitle="",
        legend=("Bx", "By", "Bz", "Bt"),
        line_color=("b", "g", "r", "k"),
    )

    # figure and axes
    fig, axs = plt.subplots(8, 1, figsize=(24, 18), sharex=True)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.15)

    # title
    tc = trange[0] + 0.5 * (trange[1] - trange[0])
    title = pd.to_datetime(tc).strftime("MMS{:1d} Wave Analysis at %Y-%m-%d %H:%M:%S".format(sc))
    fig.suptitle(title, fontsize=fontsize)

    # suppress UserWarning in agg backend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytplot.tlimit([t.strftime("%Y-%m-%d %H:%M:%S") for t in trange])
        pytplot.tplot_options("axis_font_size", fontsize)
        pytplot.tplot_opt_glob["xmargin"] = [0.05, 0.05]
        pytplot.tplot_opt_glob["ymargin"] = [0.05, 0.05]
        pytplot.tplot_opt_glob["vertical_spacing"] = 0.15
        pytplot.tplot(tplot_vars, fig=fig, axis=axs)

    # FGM and Power
    for ax in axs[0:2]:
        plt.sca(ax)
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=fontsize)

    # spectrogram
    for ax in axs[2:]:
        fce = result["fce"]
        plt.sca(ax)
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
    axs[7].set_xlabel("UT", fontsize=fontsize)

    # save file and close figure
    fig.savefig(os.sep.join([dirname, PNG_FILENAME.format(suffix, sc)]))
    plt.close(fig)


def save_parameters(sc, result, filename):
    # prepare for saving hdf5
    for key, val in result.items():
        name = "mms{:1d}_{}".format(sc, key)
        if hasattr(val, "name"):  # DataArray object
            val.name = name
        elif "name" in val:  # dict object
            val["name"] = name

    import download

    download.save_hdf5(filename, list(result.values()), append=True)

    return result


def preprocess():
    # check data availability
    fgm = [0] * 4
    scm = [0] * 4
    edp = [0] * 4
    available = [True] * 4
    for i in range(4):
        fgm[i] = pytplot.data_quants.get(FGM_FMT.format(i + 1), None)
        scm[i] = pytplot.data_quants.get(SCM_FMT.format(i + 1), None)
        edp[i] = pytplot.data_quants.get(EDP_FMT.format(i + 1), None)
        if fgm[i] is None or scm[i] is None or edp[i] is None:
            print("ignoring MMS{} data as it is not available".format(i + 1))
            available[i] = False
            continue
        # take care of duplicated times appear in some cases
        fgm[i] = fgm[i].drop_duplicates(dim="time", keep="first")
        scm[i] = scm[i].drop_duplicates(dim="time", keep="first")
        edp[i] = edp[i].drop_duplicates(dim="time", keep="first")

    # interpolation
    tindex = np.unique(scm[available.index(True)].time.values)
    for i in range(4):
        if available[i]:
            scm[i] = scm[i].interp(time=tindex, method="linear")
            edp[i] = edp[i].interp(time=tindex, method="linear")

    return dict(fgm=fgm, scm=scm, edp=edp)


def analyze_interval(trange, analyzer, dirname):
    import pytplot

    if not (os.path.exists(dirname) and os.path.isdir(dirname)):
        print("ignoring {} as it is not a directory".format(dirname))
        return

    print("processing directory: {} ...".format(dirname))

    ## load data
    import download

    download.load_hdf5(os.sep.join([dirname, "brst.h5"]), tplot=True)

    ## preprocess data
    data_dict = preprocess()

    ## try to determine shock parameters
    result = analyzer(trange, data_dict, dirname)

    ## clear
    pytplot.del_data()

    return result


def print_options(**kwargs):
    names = ("nperseg", "noverlap", "window", "wsmooth", "nsmooth", "detrend")
    print("*** perform burstwave analysis with the following options ***")
    for name in names:
        print("{:20s} : {}".format(name, kwargs.get(name)))
    print("")


if __name__ == "__main__":
    default_args = {
        "sps_acb": 8192.0,
        "sps_ace": 8192.0,
        "sps_dcb": 128.0,
        "nperseg": 2048,
        "noverlap": 512,
        "window": "blackman",
        "wsmooth": "blackman",
        "nsmooth": 5,
        "detrend": "linear",
    }

    import argparse

    parser = argparse.ArgumentParser(description="Bust Wave Analysis Tool for MMS")
    parser.add_argument("target", nargs="+", type=str, help="target file or directory")
    parser.add_argument(
        "--nperseg",
        dest="nperseg",
        type=int,
        default=2048,
        help="number of data points per segment",
    )
    parser.add_argument(
        "--noverlap",
        dest="noverlap",
        type=int,
        default=512,
        help="number of data points for overlap between consecutive segments",
    )
    parser.add_argument(
        "--window",
        dest="window",
        type=str,
        default="blackman",
        help="window function",
    )
    parser.add_argument(
        "--wsmooth",
        dest="wsmooth",
        type=str,
        default="blackman",
        help="smoothing window function in frequency space",
    )
    parser.add_argument(
        "--nsmooth",
        dest="nsmooth",
        type=int,
        default=5,
        help="number of data points for smoothing in frequency space",
    )
    parser.add_argument(
        "--detrend",
        dest="detrend",
        type=str,
        default="linear",
        help="detrend method",
    )
    parser.add_argument(
        "--suffix",
        dest="suffix",
        type=str,
        default=None,
        help="suffix for output filename",
    )
    args = parser.parse_args()

    # default suffix
    if args.suffix is None:
        args.suffix = "{}".format(args.nperseg)

    # options for analysis
    kwargs = default_args.copy()
    for key, val in vars(args).items():
        kwargs[key] = val
    print_options(**kwargs)

    analyzer = WaveAnalyzer(**kwargs)

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
            for t1, t2 in zip(tr1, tr2):
                try:
                    dirname = t1.strftime(DIR_FMT) + "-" + t2.strftime(DIR_FMT)
                    analyze_interval([t1, t2], analyzer, dirname)
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
                t1 = pd.to_datetime(tr[0], format=DIR_FMT)
                t2 = pd.to_datetime(tr[1], format=DIR_FMT)
                analyze_interval([t1, t2], analyzer, dirname)
            except Exception as e:
                print(e)
                print("Error: perhaps unrecognized directory format?")

        else:
            print("Error: {} is not a file or directory".format(target))
