#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Shock Geometry Analysis Tool for MMS

This program tries to automatically find the best upstream and downstream intervals to
estimate the shock geometry from a given time interval that defines the shock transition
layer. As a result, the right-handed LMN coordinate system and the propagation speed of
the shock in the spacecraft frame will be estimated.

The LMN coordinate is defined as follows:
  - L : transverse to the shock normal and is parallel to the tangential magnetic field.
  - M : transverse to the shock normal and is parallel to the tangential electric field.
  - N : normal to the shock surface, positive toward radially outward.
Consequently, the N-L plane corresponds to the coplanarity plane, and M is perpendicular
to it. In other words, the motional electric field (constant in the shock rest frame) is
parallel to M direction.

The shock propagation speed is estimated based on Faraday's law (constant tangential
electric field component Em) in the shock rest frame. However, the error in the shock
speed is typically on the same order of magnitude of the estimated shock speed itself.
Therefore, great care is needed to use the estimated shock speed for further analyses.

"""

import os
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg") if __name__ == "__main__" else None
from matplotlib import pyplot as plt

DIR_FMT = "%Y%m%d_%H%M%S"
JSON_FILENAME = "shockgeometry_mms{:1d}.json"


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


class AY76Analyzer:
    """Abraham-Shrauner and Yun (1976)

    Once the instance is constructed, one may estimate the shock geometry using the algorithm
    described by Abraham-Shrauner and Yan (1976) through the __call__ method. It is based on
    the coplanarity of velocity and magnetic field vectors in the both side of the shock.
    Therefore, other parameters such as density and pressure are not required for the analysis.
    The results will be the complete right-handed orthogonal (LMN) coordinate system and the
    shock propagation velocity in the spacecraft frame.

    The method tries to search for the best upstream and downstream intervals automatically.
    AY76 proposed several different methods, all of which are mathematically equivalent but
    typically give very different estimates. Each method can give an estimate for a given pair
    of upstream and downstream data. We use their MD1 and MD2 to obtain two estimates of N vector
    for each pair. The best interval is defined as such intervals for which the two different
    estimates become the closest with each other.

    """

    def __init__(self, window, deltat):
        if type(window) == int and type(deltat) == np.timedelta64:
            self.window = window
            self.deltat = deltat
        else:
            raise ValueError("Invalid input")

    def select_interval(self, t1, t2, U, B):
        index1 = U.time.searchsorted(t1.to_numpy())
        index2 = U.time.searchsorted(t2.to_numpy())
        return index1, index2, U.values[index1 : index2 + 1], B.values[index1 : index2 + 1]

    def make_pairs(self, U1, U2, B1, B2):
        N1 = U1.shape[0]
        N2 = U2.shape[0]
        U1, U2 = np.broadcast_arrays(U1[:, None, :], U2[None, :, :])
        B1, B2 = np.broadcast_arrays(B1[:, None, :], B2[None, :, :])
        U1 = U1.reshape((N1 * N2, 3))
        U2 = U2.reshape((N1 * N2, 3))
        B1 = B1.reshape((N1 * N2, 3))
        B2 = B2.reshape((N1 * N2, 3))
        return U1, U2, B1, B2

    def get_shock_parameters(self, pairs, BB):
        U1, U2, B1, B2 = pairs
        E1 = np.cross(-U1, B1, axis=-1)
        E2 = np.cross(-U2, B2, axis=-1)
        dB = B2 - B1
        dU = U2 - U1

        # LMN coordinate
        nvec = np.cross(np.cross(BB, dU, axis=-1), dB, axis=-1)
        nvec = nvec * np.sign(nvec[..., 0] + nvec[..., 1] + nvec[..., 2])[:, None]
        nvec = nvec / np.linalg.norm(nvec, axis=-1)[:, None]
        lvec = BB - np.sum(BB * nvec, axis=-1)[:, None] * nvec
        lvec = lvec * np.sign(np.sum(BB * lvec, axis=-1))[:, None]
        lvec = lvec / np.linalg.norm(lvec, axis=-1)[:, None]
        mvec = np.cross(nvec, lvec, axis=-1)
        mvec = mvec / np.linalg.norm(mvec, axis=-1)[:, None]

        # Bl : difference of tangential component of B-field
        # Em : difference of out-of-coplanarity component of E-field
        # Vs : shock speed in s/c frame
        Bl1 = np.sum(B1 * lvec, axis=-1)
        Bl2 = np.sum(B2 * lvec, axis=-1)
        Bn1 = np.sum(B1 * nvec, axis=-1)
        Bn2 = np.sum(B2 * nvec, axis=-1)
        Ul1 = np.sum(U1 * lvec, axis=-1)
        Ul2 = np.sum(U2 * lvec, axis=-1)
        Un1 = np.sum(U1 * nvec, axis=-1)
        Un2 = np.sum(U2 * nvec, axis=-1)
        Em1 = -(Un1 * Bl1 - Ul1 * Bn1)
        Em2 = -(Un2 * Bl2 - Ul2 * Bn2)
        Vs = -(Em2 - Em1) / (Bl2 - Bl1)

        return lvec, mvec, nvec, Vs

    def get_best_interval(self, U1, U2, B1, B2):
        from scipy import signal

        N1 = U1.shape[0]
        N2 = U2.shape[0]

        U1, U2, B1, B2 = self.make_pairs(U1, U2, B1, B2)
        parameters1 = self.get_shock_parameters((U1, U2, B1, B2), B1)
        parameters2 = self.get_shock_parameters((U1, U2, B1, B2), B2)

        lvec1, mvec1, nvec1, Vs1 = parameters1
        lvec2, mvec2, nvec2, Vs2 = parameters2

        # window
        W = self.window
        w = np.ones(2 * W + 1)
        w = w[:, None] * w[None, :] / (np.sum(w) * np.sum(w))

        # mask
        mask1 = np.logical_or(np.less(np.arange(N1), W), np.greater_equal(np.arange(N1), N1 - W))
        mask2 = np.logical_or(np.less(np.arange(N2), W), np.greater_equal(np.arange(N2), N2 - W))
        mask = np.logical_or(mask1[:, None], mask2[None, :])

        # find intervals with consistent nvec
        _, n_th1, n_ph1 = xyz2sph(nvec1[..., 0], nvec1[..., 1], nvec1[..., 2])
        _, n_th2, n_ph2 = xyz2sph(nvec2[..., 0], nvec2[..., 1], nvec2[..., 2])
        del_th = (n_th2 - n_th1).reshape((N1, N2))
        del_ph = (n_ph2 - n_ph1).reshape((N1, N2))
        err_l2 = signal.convolve(del_th**2 + del_ph**2, w, mode="same")
        err_l2 = np.ma.masked_array(err_l2, mask=mask)
        index1, index2 = np.unravel_index(np.argmin(err_l2), (N1, N2))

        return index1, index2

    def get_best_estimate(self, U1, U2, B1, B2):
        lvec1, mvec1, nvec1, Vs1 = self.get_shock_parameters((U1, U2, B1, B2), B1)
        lvec2, mvec2, nvec2, Vs2 = self.get_shock_parameters((U1, U2, B1, B2), B2)

        # calculate nvec
        _, n_th1, n_ph1 = xyz2sph(nvec1[:, 0], nvec1[:, 1], nvec1[:, 2])
        _, n_th2, n_ph2 = xyz2sph(nvec2[:, 0], nvec2[:, 1], nvec2[:, 2])
        n_th = 0.5 * (n_th1 + n_th2).mean()
        n_ph = 0.5 * (n_ph1 + n_ph2).mean()
        nvec = np.array(sph2xyz(1, n_th, n_ph))
        nvec_err = np.sqrt(0.25 * (np.var(n_th1) + np.var(n_th2) + np.var(n_ph1) + np.var(n_ph2)))

        # lvec and mvec
        BB = B1.mean(axis=0)
        lvec = BB - np.sum(BB * nvec, axis=-1) * nvec
        lvec = lvec * np.sign(np.sum(BB * lvec, axis=-1))
        lvec = lvec / np.linalg.norm(lvec, axis=-1)
        mvec = np.cross(nvec, lvec, axis=-1)
        mvec = mvec / np.linalg.norm(mvec, axis=-1)

        # Vs
        Vs = 0.5 * (Vs1 + Vs2).mean()
        Vs_err = np.sqrt(0.5 * (np.var(Vs1) + np.var(Vs2)))

        return lvec, mvec, nvec, Vs, nvec_err, Vs_err

    def __call__(self, trange, data_dict, dirname, quality):
        t1, t2 = trange
        U = data_dict["vi"]
        B = data_dict["bf"]
        T = np.timedelta64(0, "s")

        # candidate left interval
        tl1 = pd.to_datetime(t1) - self.deltat
        tl2 = pd.to_datetime(t1) + T
        il1, il2, Ul, Bl = self.select_interval(tl1, tl2, U, B)

        # candidate right interval
        tr1 = pd.to_datetime(t2) - T
        tr2 = pd.to_datetime(t2) + self.deltat
        ir1, ir2, Ur, Br = self.select_interval(tr1, tr2, U, B)

        # find best interval
        index1, index2 = self.get_best_interval(Ul, Ur, Bl, Br)
        l_index = index1 - self.window + il1, index1 + self.window + il1
        r_index = index2 - self.window + ir1, index2 + self.window + ir1
        l_trange = B.time.values[l_index[0]], B.time.values[l_index[1]]
        r_trange = B.time.values[r_index[0]], B.time.values[r_index[1]]

        ul = U.values[l_index[0] : l_index[1] + 1, :]
        bl = B.values[l_index[0] : l_index[1] + 1, :]
        ur = U.values[r_index[0] : r_index[1] + 1, :]
        br = B.values[r_index[0] : r_index[1] + 1, :]

        # estiamte errors
        lvec, mvec, nvec, Vs, error_nvec, error_vshn = self.get_best_estimate(ul, ur, bl, br)

        # transformation velocity to NIF
        V = 0.5 * (ul.mean(axis=0) + ur.mean(axis=0))
        Vshock = np.array([np.dot(V, lvec), np.dot(V, mvec), Vs])

        # store result
        result = dict(
            analyzer=dict(
                name="AY76",
                sc=data_dict["sc"],
                window=self.window,
                deltat=int(self.deltat / np.timedelta64(1, "s")),
            ),
            quality=quality,
            l_index=l_index,
            r_index=r_index,
            l_trange=l_trange,
            r_trange=r_trange,
            c_trange=[ts.to_numpy() for ts in trange],
            lvec=lvec,
            mvec=mvec,
            nvec=nvec,
            Vshock=Vshock,
            error_nvec=error_nvec,
            error_vshn=error_vshn,
        )

        # calculate and save parameters
        parameters = save_parameters(data_dict, result, dirname)

        # summary plot with LMN coordinate in NIF for visual inspection
        t1 = pd.to_datetime(tl1) - np.timedelta64(1, "m")
        t2 = pd.to_datetime(tr2) + np.timedelta64(1, "m")
        figure = plot_summary_lmn([t1, t2], data_dict, result, parameters, dirname)

        return result, figure


def plot_summary_lmn(trange, data_dict, result, parameters, dirname):
    import matplotlib as mpl
    from matplotlib import pylab as plt
    import pytplot
    import utils

    lvec = result["lvec"]
    mvec = result["mvec"]
    nvec = result["nvec"]
    LMN = np.vstack([lvec, mvec, nvec])[None, :, :]
    Vshock = result["Vshock"]

    sc = data_dict["sc"]
    ne = data_dict["ne"].values
    ni = data_dict["ni"].values
    bf = data_dict["bf"].values
    vi = data_dict["vi"].values
    tt = data_dict["bf"].time.values

    # convert to LMN coordinate in NIF
    bf_lmn = np.sum(LMN * bf[:, None, :], axis=-1)
    vi_lmn = np.sum(LMN * vi[:, None, :], axis=-1) - Vshock[None, :]
    ef_lmn = np.cross(bf_lmn, vi_lmn, axis=-1) * 1.0e-3

    # store data to plot
    density = np.vstack([ni, ne]).swapaxes(0, 1)
    pytplot.store_data("density", data=dict(x=tt, y=density))
    pytplot.store_data("bf_lmn", data=dict(x=tt, y=bf_lmn))
    pytplot.store_data("ef_lmn", data=dict(x=tt, y=ef_lmn))
    pytplot.store_data("vi_lmn", data=dict(x=tt, y=vi_lmn))

    # set plot options
    utils.set_plot_option(
        pytplot.data_quants["density"],
        ylabel=r"N [1/cm$^3$]",
        legend=("Ni", "Ne"),
        line_color=("r", "b"),
        char_size=10,
    )
    utils.set_plot_option(
        pytplot.data_quants["bf_lmn"],
        ylabel="B [nT]",
        legend=("L", "M", "N"),
        line_color=("b", "g", "r"),
        char_size=10,
    )
    utils.set_plot_option(
        pytplot.data_quants["ef_lmn"],
        ylabel="E [mV/m]",
        legend=("L", "M", "N"),
        line_color=("b", "g", "r"),
        char_size=10,
    )
    utils.set_plot_option(
        pytplot.data_quants["vi_lmn"],
        ylabel="V [km/s]",
        legend=("L", "M", "N"),
        line_color=("b", "g", "r"),
        char_size=10,
    )

    # figure
    npanels = 4
    fig, axs = plt.subplots(
        nrows=npanels, sharex=True, gridspec_kw={"height_ratios": npanels * [1]}
    )
    fig.set_size_inches(8, 8)

    # suppress UserWarning in agg backend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytplot.tlimit([t.strftime("%Y-%m-%d %H:%M:%S") for t in trange])
        pytplot.tplot_options("axis_font_size", 12)
        pytplot.tplot(["density", "bf_lmn", "vi_lmn", "ef_lmn"], fig=fig, axis=axs)

    # customize appearance
    lxrange = [result["l_trange"][0], result["l_trange"][1]]
    rxrange = [result["r_trange"][0], result["r_trange"][1]]
    cxrange = np.logical_and(tt >= result["c_trange"][0], tt <= result["c_trange"][1])
    lyrange = [1.0, 1.0]
    ryrange = [1.0, 1.0]
    tc = result["c_trange"][0] + 0.5 * (result["c_trange"][1] - result["c_trange"][0])
    title = pd.to_datetime(tc).strftime(
        "MMS{:1d} Bow Shock at %Y-%m-%d %H:%M:%S (Normal Incidence Frame)".format(sc)
    )
    title += "\n"
    title += r"$M_{{A}}$ = {:7.3f} $\pm$ {:5.3f}; ".format(*parameters["Ma_nif_i"])
    title += r"$\cos \theta_{{Bn}}$ = {:7.3f} $\pm$ {:5.3f}; ".format(*parameters["cos_tbn"])
    title += r"$|B_0|$ = {:7.3f} $\pm$ {:5.3f}; ".format(*parameters["Bt1"])

    fig.subplots_adjust(left=0.12, right=0.88, top=0.93, bottom=0.08)
    fig.suptitle(title, fontsize=12, x=0.5, y=0.98)

    for ax in axs:
        plt.sca(ax)
        plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        plt.grid(True, linestyle="--")
        plt.plot(lxrange, lyrange, color="c", linewidth=10.0, transform=ax.get_xaxis_transform())
        plt.plot(rxrange, ryrange, color="c", linewidth=10.0, transform=ax.get_xaxis_transform())
        plt.fill_between(
            tt, 0, 1, where=cxrange, color="grey", alpha=0.20, transform=ax.get_xaxis_transform()
        )
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mpl.dates.MinuteLocator())
        ax.xaxis.set_minor_locator(mpl.dates.SecondLocator(bysecond=range(0, 60, 10)))
    axs[-1].set_xlabel("UT")
    fig.align_ylabels(axs)

    return fig


def save_parameters(data_dict, result, dirname):
    from scipy import constants
    import pyspedas
    import pytplot

    sc = data_dict["sc"]
    Ne = data_dict["ne"]
    Ni = data_dict["ni"]
    Ue = data_dict["ve"]
    Ui = data_dict["vi"]
    Pe = data_dict["pe"]
    Pi = data_dict["pi"]
    Bf = data_dict["bf"]

    l_index = result["l_index"]
    r_index = result["r_index"]
    lvec = result["lvec"]
    mvec = result["mvec"]
    nvec = result["nvec"]
    Vshock = result["Vshock"]

    # shock transition interval
    trange = np.datetime_as_string(result["c_trange"])

    # upstream and downstream indices
    index1 = np.arange(l_index[0], l_index[1] + 1)
    index2 = np.arange(r_index[0], r_index[1] + 1)
    trange1 = np.datetime_as_string(result["l_trange"])
    trange2 = np.datetime_as_string(result["r_trange"])

    # swap for outbound crossing
    if Ne.values[index1].mean() > Ne.values[index2].mean():
        index1, index2 = index2, index1
        trange1, trange2 = trange2, trange1

    ## take average and standard deviation
    stdmean = lambda f: (
        np.mean(f, axis=0),
        np.std(f, axis=0),
    )

    # upstream
    Ne1_avg, Ne1_err = stdmean(Ne.values[index1])
    Ni1_avg, Ni1_err = stdmean(Ni.values[index1])
    Ue1_avg, Ue1_err = stdmean(Ue.values[index1, :])
    Ui1_avg, Ui1_err = stdmean(Ui.values[index1, :])
    Pe1_avg, Pe1_err = stdmean(Pe.values[index1])
    Pi1_avg, Pi1_err = stdmean(Pi.values[index1])
    Bf1_avg, Bf1_err = stdmean(Bf.values[index1, :])
    # downstream
    Ne2_avg, Ne2_err = stdmean(Ne.values[index2])
    Ni2_avg, Ni2_err = stdmean(Ni.values[index2])
    Ue2_avg, Ue2_err = stdmean(Ue.values[index2, :])
    Ui2_avg, Ui2_err = stdmean(Ui.values[index2, :])
    Pe2_avg, Pe2_err = stdmean(Pe.values[index2])
    Pi2_avg, Pi2_err = stdmean(Pi.values[index2])
    Bf2_avg, Bf2_err = stdmean(Bf.values[index2, :])

    Un1 = np.dot(Ui1_avg, nvec)
    Un2 = np.dot(Ui2_avg, nvec)
    Bl1 = np.dot(Bf1_avg, lvec)
    Bm1 = np.dot(Bf1_avg, mvec)
    Bn1 = np.dot(Bf1_avg, nvec)
    Bt1 = np.linalg.norm(Bf1_avg)
    Bt1_err = np.sqrt(Bf1_err[0] ** 2 + Bf1_err[1] ** 2 + Bf1_err[2] ** 2)

    # shock obliquity
    theta_bn = np.rad2deg(np.arctan2(Bl1, Bn1))
    theta_bn_err = result["error_nvec"]
    cos_tbn = Bn1 / Bt1
    cos_tbn_err = np.sqrt(
        (1 - cos_tbn**2) * np.deg2rad(theta_bn_err) ** 2 + np.dot(Bf1_err / Bt1, nvec) ** 2
    )

    # shock speed
    Vs_n_scf = Vshock[2]
    Vs_n_scf_err = result["error_vshn"]
    Vs_n_nif = Un1 - Vs_n_scf
    Vs_n_nif_err = Vs_n_scf_err

    # upstream density correction
    Ne1_crct = Ne2_avg * (Un2 - Vshock[2]) / (Un1 - Vshock[2])
    Ne1_crct_err = Ne2_err * (Un2 - Vshock[2]) / (Un1 - Vshock[2])
    Ni1_crct = Ni2_avg * (Un2 - Vshock[2]) / (Un1 - Vshock[2])
    Ni1_crct_err = Ni2_err * (Un2 - Vshock[2]) / (Un1 - Vshock[2])

    # Mach number
    Vs_abs = np.abs(Vs_n_nif)
    Vs_err = Vs_n_scf_err
    Ma_nif_i = 4.586e-2 * Vs_abs * np.sqrt(Ni1_avg) / Bt1
    Ma_nif_i_err = Ma_nif_i * np.sqrt(
        (Vs_err / Vs_abs) ** 2 + (Bt1_err / Bt1) ** 2 + 0.25 * (Ni1_err / Ni1_avg) ** 2
    )
    Ma_nif_e = 4.586e-2 * Vs_abs * np.sqrt(Ne1_avg) / Bt1
    Ma_nif_e_err = Ma_nif_e * np.sqrt(
        (Vs_err / Vs_abs) ** 2 + (Bt1_err / Bt1) ** 2 + 0.25 * (Ne1_err / Ne1_avg) ** 2
    )

    # pressure plasma beta
    Pb = (np.linalg.norm(Bf1_avg) ** 2 / (2 * constants.mu_0)) * 1e-9

    # average OMNI data
    omni_time_range = slice(
        result["l_trange"][0] - np.timedelta64(30, "m"),
        result["r_trange"][1] + np.timedelta64(30, "m"),
    )
    omni_stdmean = lambda f: (
        f.sel(time=omni_time_range).mean().item(),
        f.sel(time=omni_time_range).std().item(),
    )
    Ni_omni = pytplot.get_data("omni_ni", xarray=True)
    Ma_omni = pytplot.get_data("omni_mach", xarray=True)
    Beta_omni = pytplot.get_data("omni_beta", xarray=True)

    parameters = {
        "analyzer": result["analyzer"],
        "quality": result["quality"],
        "trange": list(trange),
        "trange1": list(trange1),
        "trange2": list(trange2),
        # upstream
        "Ne1": (Ne1_avg, Ne1_err),
        "Ni1": (Ni1_avg, Ni1_err),
        "Uex1": (Ue1_avg[0], Ue1_err[0]),
        "Uey1": (Ue1_avg[1], Ue1_err[1]),
        "Uez1": (Ue1_avg[2], Ue1_err[2]),
        "Uix1": (Ui1_avg[0], Ui1_err[0]),
        "Uiy1": (Ui1_avg[1], Ui1_err[1]),
        "Uiz1": (Ui1_avg[2], Ui1_err[2]),
        "Pe1": (Pe1_avg, Pe1_err),
        "Pi1": (Pi1_avg, Pi1_err),
        "Bx1": (Bf1_avg[0], Bf1_err[0]),
        "By1": (Bf1_avg[1], Bf1_err[1]),
        "Bz1": (Bf1_avg[2], Bf1_err[2]),
        # downstream
        "Ne2": (Ne2_avg, Ne2_err),
        "Ni2": (Ni2_avg, Ni2_err),
        "Uex2": (Ue2_avg[0], Ue2_err[0]),
        "Uey2": (Ue2_avg[1], Ue2_err[1]),
        "Uez2": (Ue2_avg[2], Ue2_err[2]),
        "Uix2": (Ui2_avg[0], Ui2_err[0]),
        "Uiy2": (Ui2_avg[1], Ui2_err[1]),
        "Uiz2": (Ui2_avg[2], Ui2_err[2]),
        "Pe2": (Pe2_avg, Pe2_err),
        "Pi2": (Pi2_avg, Pi2_err),
        "Bx2": (Bf2_avg[0], Bf2_err[0]),
        "By2": (Bf2_avg[1], Bf2_err[1]),
        "Bz2": (Bf2_avg[2], Bf2_err[2]),
        # shock parameters
        "Bt1": (Bt1, Bt1_err),
        "Ne1_crct": (Ne1_crct, Ne1_crct_err),
        "Ni1_crct": (Ni1_crct, Ni1_crct_err),
        "theta_bn": (theta_bn, theta_bn_err),
        "cos_tbn": (cos_tbn, cos_tbn_err),
        "Vs_n_nif": (Vs_n_nif, Vs_n_nif_err),
        "Vs_n_scf": (Vs_n_scf, Vs_n_scf_err),
        "Ma_nif_i": (Ma_nif_i, Ma_nif_i_err),
        "Ma_nif_e": (Ma_nif_e, Ma_nif_e_err),
        "Beta_i": (Pi1_avg / Pb, Pi1_err / Pb),
        "Beta_e": (Pe1_avg / Pb, Pe1_err / Pb),
        # coordinate
        "lvec": list(lvec),
        "mvec": list(mvec),
        "nvec": list(nvec),
        # OMNI
        "Ni_omni": omni_stdmean(Ni_omni),
        "Ma_omni": omni_stdmean(Ma_omni),
        "Beta_omni": omni_stdmean(Beta_omni),
    }

    print("{:20s} : {}".format("trange1", parameters["trange1"]))
    print("{:20s} : {}".format("trange2", parameters["trange2"]))
    keywords = (
        "Ne1",
        "Ne1_crct",
        "Ni1",
        "Ni1_crct",
        "Bt1",
        "theta_bn",
        "cos_tbn",
        "Vs_n_scf",
        "Vs_n_nif",
        "Ma_nif_i",
        "Ma_nif_e",
        "Beta_i",
        "Beta_e",
        "Ni_omni",
        "Ma_omni",
        "Beta_omni",
    )
    for key in keywords:
        if isinstance(parameters[key], tuple):
            args = (key,) + parameters[key]
            print("{:20s} : {:10.4f} +- {:5.2f}".format(*args))
        else:
            print("{:20s} : {:10.4f}".format(key, +parameters[key]))

    with open(os.sep.join([dirname, JSON_FILENAME.format(sc)]), "w") as fp:
        fp.write(json.dumps(parameters, indent=4))

    return parameters


def preprocess():
    import pytplot
    import utils

    vardict = pytplot.data_quants
    Bf = [0] * 4
    Ni = [0] * 4
    Ne = [0] * 4
    Vi = [0] * 4
    Ve = [0] * 4
    Pi = [0] * 4
    Pe = [0] * 4
    for i in range(4):
        sc = "mms%d_" % (i + 1)
        Bf[i] = vardict[sc + "fgm_b_gse_srvy_l2"]
        Ni[i] = vardict[sc + "dis_numberdensity_fast"]
        Ne[i] = vardict[sc + "des_numberdensity_fast"]
        Vi[i] = vardict[sc + "dis_bulkv_gse_fast"]
        Ve[i] = vardict[sc + "des_bulkv_gse_fast"]
        Pi[i] = vardict[sc + "dis_prestensor_gse_fast"]
        Pe[i] = vardict[sc + "des_prestensor_gse_fast"]

    #
    # (1) downsample magnetic field
    # (2) interpolate moment quantities
    #
    dt = 4.5 * 1.0e9
    tt = Bf[0].time.values
    nn = (tt[-1] - tt[0]) / dt
    tb = np.arange(nn) * dt + tt[0]
    tc = tb[:-1] + 0.5 * (tb[+1:] - tb[:-1])

    data = [0] * 4
    for i in range(4):
        sc = i + 1
        bf = utils.create_xarray(x=tc, y=np.zeros((tc.size, 3)))
        ni = utils.create_xarray(x=tc, y=np.zeros((tc.size,)))
        ne = utils.create_xarray(x=tc, y=np.zeros((tc.size,)))
        vi = utils.create_xarray(x=tc, y=np.zeros((tc.size, 3)))
        ve = utils.create_xarray(x=tc, y=np.zeros((tc.size, 3)))
        pi = utils.create_xarray(x=tc, y=np.zeros((tc.size,)))
        pe = utils.create_xarray(x=tc, y=np.zeros((tc.size,)))
        bf[...] = Bf[i].groupby_bins("time", tb).mean().values[:, 0:3]
        ni[...] = Ni[i].interp(time=tc).values
        ne[...] = Ne[i].interp(time=tc).values
        vi[...] = Vi[i].interp(time=tc).values
        ve[...] = Ve[i].interp(time=tc).values
        pi[...] = np.trace(Pi[i].interp(time=tc).values, axis1=1, axis2=2) / 3
        pe[...] = np.trace(Pe[i].interp(time=tc).values, axis1=1, axis2=2) / 3
        # ignore if NaN is detected in any elements except for first and last
        for x in bf, ni, ne, vi, ve, pi, pe:
            if np.any(np.isnan(x[+1:-1])) == True:
                available = False
            else:
                available = True
        data[i] = dict(bf=bf, ni=ni, ne=ne, vi=vi, ve=ve, pi=pi, pe=pe, sc=sc, available=available)

    return data


def analyze_interval(trange, analyzer, dirname, quality=1):
    import pytplot

    if not (os.path.exists(dirname) and os.path.isdir(dirname)):
        print("ignoreing {} as it is not a directory".format(dirname))
        return

    ## load data
    import download

    download.load_hdf5(os.sep.join([dirname, "fast.h5"]), tplot=True)
    download.load_hdf5(os.sep.join([dirname, "omni.h5"]), tplot=True)

    ## preprocess data
    data_dict = preprocess()

    ## try to determine shock parameters
    for i in range(4):
        if data_dict[i]["available"]:
            result, figure = analyzer(trange, data_dict[i], dirname, quality)
            sc = data_dict[i]["sc"]
            figure.savefig(os.sep.join([dirname, "summary_lmn_nif_mms{:1d}.png".format(sc)]))
        else:
            print("MMS{:1d} data is not available for analysis".format(i + 1))

    ## clear
    pytplot.del_data()

    return result


def json2dict(js, ID):
    # ID
    d = dict(ID=ID)

    # event quality
    d["quality"] = js["quality"]

    # interval
    d["interval_1"] = "{}".format(js["trange"][0])
    d["interval_2"] = "{}".format(js["trange"][1])
    d["u_interval_1"] = "{}".format(js["trange1"][0])
    d["u_interval_2"] = "{}".format(js["trange1"][1])
    d["d_interval_1"] = "{}".format(js["trange2"][0])
    d["d_interval_2"] = "{}".format(js["trange2"][1])

    # LMN coordinate
    for key in ("lvec", "mvec", "nvec"):
        d["{}.x".format(key)] = js[key][0]
        d["{}.y".format(key)] = js[key][1]
        d["{}.z".format(key)] = js[key][2]

    # add arbitrary parameters
    ignore_keys = (
        "trange",
        "trange1",
        "trange2",
        "lvec",
        "mvec",
        "nvec",
        "analyzer",
        "quality",
        "sc",
    )
    for key in js.keys():
        if key in ignore_keys:
            continue
        # assume (average, error) pair
        d["{}_avg".format(key)] = js[key][0]
        d["{}_err".format(key)] = js[key][1]

    # meta data
    for key, item in js["analyzer"].items():
        d["analyzer.{}".format(key)] = item

    return d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shock Geometry Analysis Tool for MMS")
    parser.add_argument("target", nargs="+", type=str, help="target file or directory")
    parser.add_argument(
        "-w",
        "--window",
        dest="window",
        type=int,
        default=3,
        help="upstream/downstream interval size in data points",
    )
    parser.add_argument(
        "-d",
        "--deltat",
        dest="deltat",
        type=int,
        default=90,
        help="time range for the best time interval search in second",
    )
    args = parser.parse_args()

    # analyzer
    analyzer = AY76Analyzer(args.window, np.timedelta64(args.deltat, "s"))

    ###
    ### analyze for each target
    ###
    import download

    targetlist = []
    for target in args.target:
        if os.path.isfile(target):
            #
            # event list file in CSV format
            #
            tr1, tr2 = download.read_eventlist(target)
            csv = pd.read_csv(target, header=None, skiprows=1)
            tr1 = pd.to_datetime(csv.iloc[:, 0])
            tr2 = pd.to_datetime(csv.iloc[:, 1])
            quality = csv.iloc[:, 2]
            for t1, t2, q in zip(tr1, tr2, quality):
                try:
                    dirname = t1.strftime(DIR_FMT) + "-" + t2.strftime(DIR_FMT)
                    analyze_interval([t1, t2], analyzer, dirname, q)
                    targetlist.append(dirname)
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
                targetlist.append(dirname)
            except Exception as e:
                print(e)
                print("Error: perhaps unrecognized directory format?")

        else:
            print("Error: {} is not a file or directory".format(target))
