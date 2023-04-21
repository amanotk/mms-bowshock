# -*- coding: utf-8 -*-

""" Wave analysis tools

References
----------
- Santolik et al., J. Geophys. Res., 107, 1444, 2002
- Santolik et al., Rdaio Sci., 38(1), 1010, 2003
- Santolik et al., J. Geophys. Res., 115, A00F13, 2010
"""

import numpy as np
from numpy import ma
import scipy as sp
from scipy import fftpack, signal, ndimage, constants

import xarray as xr
import pandas as pd

from utils import cast_list
from utils import set_plot_option
from utils import get_default_tplot_attrs


def get_default_spectrogram_attrs():
    attrs = get_default_tplot_attrs()
    attrs["plot_options"]["yaxis_opt"]["y_axis_type"] = "log"
    attrs["plot_options"]["yaxis_opt"]["axis_label"] = "Freq [Hz]"
    attrs["plot_options"]["extras"]["spec"] = True
    attrs["plot_options"]["spec_bins_ascending"] = True
    return attrs


def get_mfa_unit_vector(bx, by, bz):
    """Calculate unit vectors for Magnetic-Field-Aligned coordinate

    e1 : perpendicular to B and lies in the x-z plane
    e2 : e3 x e1
    e3 : parallel to B

    Parameters
    ----------
    bx, by, bz : array-like
        three components of magnetic field

    Returns
    -------
    e1, e2, e3 : array-like
        unit vectors
    """
    bx = np.atleast_1d(bx)
    by = np.atleast_1d(by)
    bz = np.atleast_1d(bz)
    bb = np.sqrt(bx**2 + by**2 + bz**2) + 1.0e-32
    sh = bb.shape + (3,)
    e1 = np.zeros(sh, np.float64)
    e2 = np.zeros(sh, np.float64)
    e3 = np.zeros(sh, np.float64)
    # e3 parallel to B
    e3[..., 0] = bx / bb
    e3[..., 1] = by / bb
    e3[..., 2] = bz / bb
    # e1 is perpendicular to B and in x-z plane
    e1z = -e3[..., 0] / (e3[..., 2] + 1.0e-32)
    e1[..., 0] = 1.0 / np.sqrt(1.0 + e1z**2)
    e1[..., 1] = 0.0
    e1[..., 2] = e1z / np.sqrt(1.0 + e1z**2)
    # e2 = e3 x e1
    e2 = np.cross(e3, e1, axis=-1)
    # back to scalar
    if bx.size == 1 and by.size == 1 and bz.size == 1:
        e1 = e1[0, :]
        e2 = e2[0, :]
        e3 = e3[0, :]
    return e1, e2, e3


def transform_vector(vx, vy, vz, e1, e2, e3):
    """Transform vector (vx, vy, vz) to given coordinate system (e1, e2, e3)

    Parameters
    ----------
    vx, vy, vz : array-like
        input vector
    e1, e2, e3 : array-like
        unit vectors for the new coordinate system

    Returns
    -------
    v1, v2, v3 : array-like
        each vector component in the new coordinate system
    """
    if e1.ndim == 1 and e2.ndim == 1 and e2.ndim == 1:
        v1 = vx * e1[0] + vy * e1[1] + vz * e1[2]
        v2 = vx * e2[0] + vy * e2[1] + vz * e2[2]
        v3 = vx * e3[0] + vy * e3[1] + vz * e3[2]
    else:
        v1 = vx * e1[:, None, 0] + vy * e1[:, None, 1] + vz * e1[:, None, 2]
        v2 = vx * e2[:, None, 0] + vy * e2[:, None, 1] + vz * e2[:, None, 2]
        v3 = vx * e3[:, None, 0] + vy * e3[:, None, 1] + vz * e3[:, None, 2]
    return v1, v2, v3


def segmentalize(x, nperseg, noverlap):
    """Segmentalize the input array

    This may be useful for custom spectrogram calculation.
    See: scipy.signal.spectral._fft_helper

    Parameters
    ----------
    x : array-like
        input array
    nperseg : int
        data size for each segment
    noverlap : int
        data size for for overlap interval (default nperseg/2)

    Returns
    -------
    segmentalized data
    """
    step = nperseg - noverlap
    sh = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    st = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=sh, strides=st, writeable=False)
    return result


def spectrogram(x, fs, nperseg, noverlap=None, window=None, detrend=None):
    """Calculate power spectral density spectrogram

    Parameters
    ----------
    x : array-like or list of array-like
        time series data
    fs : float
        sampling frequency
    nperseg : int
        number of data points for each segment
    noverlap : int
        number of overlapped data points (nperseg/2 by default)
    window : str
        window applied for each segment ('blackman' is used by default)
    detrend : str
        detrend method (either 'constant' or 'linear')

    Returns
    -------
    If x is xarray's DataArray object, result will also be returned as a
    DataArray object. Otherwise, frequecy, time, power spectral density will be
    returned as a tuple.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if window is None:
        window = "blackman"
    if detrend is None:
        detrend = False

    args = {
        "fs": fs,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "window": window,
        "detrend": detrend,
    }

    # calculate sum of all input
    x = cast_list(x)
    f, t, s = signal.spectrogram(x[0], **args)
    for i in range(1, len(x)):
        ff, tt, ss = signal.spectrogram(x[i], **args)
        if s.shape == ss.shape:
            s[...] = s[...] + ss
        else:
            raise ValueError("Invalid input data")

    # discard zero frequency
    f = f[1:]
    s = s[1:, :]

    # return xarray if input is xarray
    is_xarray = np.all([isinstance(xx, xr.DataArray) for xx in x])
    if is_xarray:
        t = pd.to_datetime(t + x[0].time.values[0].astype(np.int64) * 1e-9, unit="s")
        s = s.transpose()
        f = np.repeat(f[np.newaxis, :], s.shape[0], axis=0)
        bins = xr.DataArray(f, dims=("time", "f"), coords={"time": t})

        # DataArray
        args = {
            "dims": ("time", "f"),
            "coords": {"time": t, "spec_bins": bins},
        }
        data = xr.DataArray(s, **args)

        # set attribute
        data.attrs = get_default_spectrogram_attrs()
        set_plot_option(data, yrange=[f[0, 0], f[0, -1]], trange=[t[0], t[-1]], z_type="log")

        return data
    else:
        # otherwise simple sum of all spectra
        return f, t, s


def msvd(ace, acb, dcb, **kwargs):
    """Perform wave polarization analysis via magnetic SVD

    This is a convinient wrapper function using MSVD class.

    Parameters
    ----------
    ace : DataArray object
        high-frequency three-component electric field data
    acb : DataArray object
        high-frequency three-component magnetic field data
    acb : DataArray object
        low-frequency three-component magnetic field data
    sps_ace : int or float
        sample per second for ace (8192 by default)
    sps_acb : int or float
        sample per second for acb (8192 by default)
    sps_dcb : int or float
        sample per second for dcb (128 by default)
    nperseg : int
        number of data points for each segment for fft (1024 by default)
    noverlap : int
        number of data points for neighboring segment overlap (nperseg/2 by
        default)
    window  : str
        window function ('blackman' by default)
    wsmooth : str
        window function for smoothing spectral matrix in time ('blackman' by
        default)
    nsmooth : int
        number of data points for smoothing window (5 by default)
    detrend : str
        detrending method for each segment. one of 'constant', 'linear', or
        False. (False by default)

    Returns
    -------
    dictionary of analysis results. each item is a DataArray object which can be
    displayed as aspectrogram.
    """
    svd = MSVD(**kwargs)
    return svd.analyze(ace, acb, dcb)


class MSVD:
    """Magnetic Singular Value Decomposition Method"""

    def __init__(self, **kwargs):
        default_args = {
            "sps_acb": 8192.0,
            "sps_ace": 8192.0,
            "sps_dcb": 128.0,
            "nperseg": 1024,
            "noverlap": 512,
            "window": "blackman",
            "wsmooth": "blackman",
            "nsmooth": 5,
            "detrend": False,
        }
        for key in default_args.keys():
            setattr(self, key, kwargs.get(key, default_args[key]))

    def calc_mfa_coord(self, dcb, ti, nperseg, noverlap):
        # magnetic field averaged over given segments
        bb = dcb.interp(time=ti, method="linear")
        bx = segmentalize(bb.values[:, 0], nperseg, noverlap).mean(axis=1)
        by = segmentalize(bb.values[:, 1], nperseg, noverlap).mean(axis=1)
        bz = segmentalize(bb.values[:, 2], nperseg, noverlap).mean(axis=1)
        bt = np.sqrt(bx**2 + by**2 + bz**2)
        return get_mfa_unit_vector(bx / bt, by / bt, bz / bt)

    def spectral_matrix(self, ace, acb, dcb):
        # calculate spectral matrix
        convolve = ndimage.convolve1d
        sps_acb = float(self.sps_acb)
        sps_dcb = float(self.sps_dcb)
        nperseg = self.nperseg
        noverlap = self.noverlap
        window = self.window
        nsmooth = self.nsmooth
        wsmooth = self.wsmooth
        nsegment = nperseg - noverlap
        nfreq = nperseg // 2

        # data
        nt = acb.shape[0]
        ti = acb.time.values[:]
        bx = acb.values[:, 0]
        by = acb.values[:, 1]
        bz = acb.values[:, 2]
        ef = ace.interp(time=ti, method="linear")
        ex = ef.values[:, 0]
        ey = ef.values[:, 1]
        ez = ef.values[:, 2]
        ww = signal.get_window(window, nperseg)
        ww = ww / ww.sum()
        mt = (nt - noverlap) // nsegment

        # segmentalize
        Bx = segmentalize(bx, nperseg, noverlap) * ww[None, :]
        By = segmentalize(by, nperseg, noverlap) * ww[None, :]
        Bz = segmentalize(bz, nperseg, noverlap) * ww[None, :]
        Ex = segmentalize(ex, nperseg, noverlap) * ww[None, :]
        Ey = segmentalize(ey, nperseg, noverlap) * ww[None, :]
        Ez = segmentalize(ez, nperseg, noverlap) * ww[None, :]
        # time and frequency coordinate
        tt = pd.to_datetime(
            segmentalize(ti.astype(np.int64), nperseg, noverlap).mean(axis=1), unit="ns"
        )
        ff = np.arange(1, nfreq + 1) / (nperseg / sps_acb)

        # coordinate transformation and FFT (discard zero frequency)
        e1, e2, e3 = self.calc_mfa_coord(dcb, ti, nperseg, noverlap)
        B1, B2, B3 = transform_vector(Bx, By, Bz, e1, e2, e3)
        E1, E2, E3 = transform_vector(Ex, Ey, Ez, e1, e2, e3)
        B1 = fftpack.fft(B1, axis=-1)[:, 1 : nfreq + 1]
        B2 = fftpack.fft(B2, axis=-1)[:, 1 : nfreq + 1]
        B3 = fftpack.fft(B3, axis=-1)[:, 1 : nfreq + 1]
        E1 = fftpack.fft(E1, axis=-1)[:, 1 : nfreq + 1]
        E2 = fftpack.fft(E2, axis=-1)[:, 1 : nfreq + 1]
        E3 = fftpack.fft(E3, axis=-1)[:, 1 : nfreq + 1]

        # calculate 3x3 spectral matrix with smoothing
        ss = 1 / (sps_acb * (ww * ww).sum())  # PSD in units of nT^2/Hz
        ws = signal.get_window(wsmooth, nsmooth)
        ws = ws / ws.sum()
        sma = 0  # smoothing along time
        Q00 = B1 * np.conj(B1) * ss
        Q01 = B1 * np.conj(B2) * ss
        Q02 = B1 * np.conj(B3) * ss
        Q11 = B2 * np.conj(B2) * ss
        Q12 = B2 * np.conj(B3) * ss
        Q22 = B3 * np.conj(B3) * ss
        Q00_re = convolve(Q00.real, ws, mode="nearest", axis=sma)
        Q00_im = convolve(Q00.imag, ws, mode="nearest", axis=sma)
        Q01_re = convolve(Q01.real, ws, mode="nearest", axis=sma)
        Q01_im = convolve(Q01.imag, ws, mode="nearest", axis=sma)
        Q02_re = convolve(Q02.real, ws, mode="nearest", axis=sma)
        Q02_im = convolve(Q02.imag, ws, mode="nearest", axis=sma)
        Q11_re = convolve(Q11.real, ws, mode="nearest", axis=sma)
        Q11_im = convolve(Q11.imag, ws, mode="nearest", axis=sma)
        Q12_re = convolve(Q12.real, ws, mode="nearest", axis=sma)
        Q12_im = convolve(Q12.imag, ws, mode="nearest", axis=sma)
        Q22_re = convolve(Q22.real, ws, mode="nearest", axis=sma)
        Q22_im = convolve(Q22.imag, ws, mode="nearest", axis=sma)

        # real representation (6x3) for spectral matrix
        N = B1.shape[0]
        M = B1.shape[1]
        S = np.zeros((N, M, 6, 3), np.float64)
        S[:, :, 0, 0] = Q00_re
        S[:, :, 0, 1] = Q01_re
        S[:, :, 0, 2] = Q02_re
        S[:, :, 1, 0] = Q01_re
        S[:, :, 1, 1] = Q11_re
        S[:, :, 1, 2] = Q12_re
        S[:, :, 2, 0] = Q02_re
        S[:, :, 2, 1] = Q12_re
        S[:, :, 2, 2] = Q22_re
        S[:, :, 3, 0] = 0
        S[:, :, 3, 1] = Q01_im
        S[:, :, 3, 2] = Q02_im
        S[:, :, 4, 0] = -Q01_im
        S[:, :, 4, 1] = 0
        S[:, :, 4, 2] = Q12_im
        S[:, :, 5, 0] = -Q02_im
        S[:, :, 5, 1] = -Q12_im
        S[:, :, 5, 2] = 0

        # Poynting vector: E [mV/m] * B [nT] => conversion factor = 1e-12
        mu0 = constants.mu_0
        P1 = (E2 * np.conj(B3) - E3 * np.conj(B2)).real / mu0 * 1e-12
        P2 = (E3 * np.conj(B1) - E1 * np.conj(B3)).real / mu0 * 1e-12
        P3 = (E1 * np.conj(B2) - E2 * np.conj(B1)).real / mu0 * 1e-12
        p1 = convolve(P1, ws, mode="nearest", axis=sma)
        p2 = convolve(P2, ws, mode="nearest", axis=sma)
        p3 = convolve(P3, ws, mode="nearest", axis=sma)
        P = np.zeros((N, M, 3), np.float64)
        P[:, :, 0] = p1
        P[:, :, 1] = p2
        P[:, :, 2] = p3

        return tt, ff, S, P

    def svd(self, ace, acb, dcb):
        # calculate spectral matrix and Poyinting vector
        t, f, S, P = self.spectral_matrix(ace, acb, dcb)

        # perform SVD only for valid data
        N, M, _, _ = S.shape
        T = S.reshape(N * M, 6, 3)
        I = np.argwhere(np.isfinite(np.sum(T, axis=(-2, -1))))[:, 0]
        UU = np.zeros((N * M, 6, 6), np.float64)
        WW = np.zeros((N * M, 3), np.float64)
        VV = np.zeros((N * M, 3, 3), np.float64)
        UU[I], WW[I], VV[I] = np.linalg.svd(T[I])
        U = UU.reshape(N, M, 6, 6)
        W = WW.reshape(N, M, 3)
        V = VV.reshape(N, M, 3, 3)
        self.svd_result = dict(t=t, f=f, S=S, U=U, W=W, V=V, P=P)

        return t, f, self._process_svd_result(S, U, W, V, P)

    def _process_svd_result(self, S, U, W, V, P):
        eps = 1.0e-15
        Tr = lambda x: np.trace(x, axis1=2, axis2=3)
        SS = S[..., 0:3, 0:3] + S[..., 3:6, 0:3] * 1j

        r = dict()

        ### power spectral density (need to double except for Nyquist freq.)
        psd = 2 * Tr(np.abs(SS))
        if psd.shape[0] % 2 == 0:
            psd[-1, :] *= 0.5
        r["psd"] = psd

        ### degree of polarization
        ss1 = Tr(np.matmul(SS, SS))
        ss2 = Tr(SS) ** 2 + eps
        ss2 = np.where(np.isfinite(ss2), ss2, np.inf)
        r["degpol"] = 1.5 * (ss1 / ss2).real - 0.5

        ### planarity
        r["planarity"] = 1 - np.sqrt(W[..., 2] / (W[..., 0] + eps))

        ### ellipticity
        r["ellipticity"] = W[..., 1] / (W[..., 0] + eps) * np.sign(SS[..., 0, 1].imag)

        ### k vector
        k1 = np.sign(V[..., 2, 2]) * V[..., 2, 0]
        k2 = np.sign(V[..., 2, 2]) * V[..., 2, 1]
        k3 = np.sign(V[..., 2, 2]) * V[..., 2, 2]
        kk = np.sqrt(k1**2 + k2**2 + k3**2) + eps
        tkb = np.rad2deg(np.abs(np.arctan2(np.sqrt(k1**2 + k2**2), k3)))
        pkb = np.rad2deg(np.arctan2(k2, k1))
        r["n1"] = k1 / kk
        r["n2"] = k2 / kk
        r["n3"] = k3 / kk
        r["theta_kb"] = tkb
        r["phi_kb"] = pkb

        # Poynting vector
        p1 = P[..., 0]
        p2 = P[..., 1]
        p3 = P[..., 2]
        pp = np.sqrt(p1**2 + p2**2 + p3**2) + eps
        tsb = np.rad2deg(np.abs(np.arctan2(np.sqrt(p1**2 + p2**2), p3)))
        psb = np.rad2deg(np.arctan2(p2, p1))
        r["s1"] = p1 / pp
        r["s2"] = p2 / pp
        r["s3"] = p3 / pp
        r["theta_sb"] = tsb
        r["phi_sb"] = psb

        return r

    def _setup_arrays(self, t, f, result):
        default_args = {
            "dims": ("time", "f"),
            "coords": {
                "time": t,
                "spec_bins": ("f", f),
            },
        }

        # construct DataArray and store in dict
        dadict = dict()
        for key in result.keys():
            try:
                data = xr.DataArray(result[key], **default_args)
                data.name = key
                data.attrs = get_default_spectrogram_attrs()
                set_plot_option(data, yrange=[f[0], f[-1]], trange=[t[0], t[-1]])
                dadict[key] = data
            except Exception as e:
                print("Error in creating spectrogram for : %s" % (key))
                print(e)

        # power spectral density
        if "psd" in dadict:
            psd = dadict["psd"].values
            psd = ma.masked_where(np.isnan(psd), psd)
            ndec = 7
            zmax = 10.0 ** np.ceil(np.log10(psd.max()))
            zmin = 10.0 ** (zmax - ndec)
            colorbar_ticks = {
                "tickvals": np.linspace(zmin, zmax, ndec + 1),
                "ticktext": np.linspace(zmin, zmax, ndec + 1, dtype=np.int32),
            }
            colorbar_ticks = None
            set_plot_option(
                dadict["psd"],
                zlabel="PSD [nT^2/Hz]",
                ztype="log",
                zrange=[zmin, zmax],
                colormap=["viridis"],
                colorbar_ticks=colorbar_ticks,
            )

        # degree of polarization
        if "degpol" in dadict:
            colorbar_ticks = {
                "tickvals": np.linspace(0, +1, 5),
                "ticktext": np.linspace(0, +1, 5),
            }
            set_plot_option(
                dadict["degpol"],
                zlabel="Deg. Pol",
                zrange=[0.0, +1.0],
                colormap=["Greens"],
                colorbar_ticks=colorbar_ticks,
            )

        # planarity
        if "planarity" in dadict:
            colorbar_ticks = {
                "tickvals": np.linspace(0, +1, 5),
                "ticktext": np.linspace(0, +1, 5),
            }
            set_plot_option(
                dadict["planarity"], zlabel="Planarity", zrange=[0.0, +1.0], colormap=["Greens"]
            )

        # ellipticity
        if "ellipticity" in dadict:
            colorbar_ticks = {
                "tickvals": np.linspace(-1, +1, 5),
                "ticktext": np.linspace(-1, +1, 5),
            }
            set_plot_option(
                dadict["ellipticity"],
                zlabel="Ellipticity",
                zrange=[-1.0, +1.0],
                colormap=["bwr"],
                colorbar_ticks=colorbar_ticks,
            )

        # k vector
        for nn in ("n1", "n2", "n3"):
            colorbar_ticks = {
                "tickvals": np.linspace(-1, +1, 5),
                "ticktext": np.linspace(-1, +1, 5),
            }
            if nn in dadict:
                set_plot_option(
                    dadict[nn],
                    zlabel=nn,
                    zrange=[-1, +1],
                    colormap=["bwr"],
                    colorbar_ticks=colorbar_ticks,
                )

        if "theta_kb" in dadict:
            colorbar_ticks = {
                "tickvals": np.linspace(0, 90, 4),
                "ticktext": np.linspace(0, 90, 4),
            }
            set_plot_option(
                dadict["theta_kb"],
                zlabel="theta_kb",
                zrange=[0.0, 90.0],
                colormap=["bwr"],
                colorbar_ticks=colorbar_ticks,
            )

        if "phi_kb" in dadict:
            colorbar_ticks = {
                "tickvals": np.linspace(0, 360, 5),
                "ticktext": np.linspace(0, 360, 5),
            }
            set_plot_option(
                dadict["phi_kb"],
                zlabel="phi_kb",
                zrange=[0.0, 360.0],
                colormap=["bwr"],
                colorbar_ticks=colorbar_ticks,
            )

        # Poynting vector
        for ss in ("s1", "s2", "s3"):
            colorbar_ticks = {
                "tickvals": np.linspace(-1, +1, 5),
                "ticktext": np.linspace(-1, +1, 5),
            }
            if ss in dadict:
                set_plot_option(
                    dadict[ss],
                    zlabel=ss,
                    zrange=[-1, +1],
                    colormap=["bwr"],
                    colorbar_ticks=colorbar_ticks,
                )

        if "theta_sb" in dadict:
            colorbar_ticks = {
                "tickvals": np.linspace(0, 180, 5),
                "ticktext": np.linspace(0, 180, 5),
            }
            set_plot_option(
                dadict["theta_sb"],
                zlabel="theta_sb",
                zrange=[0.0, 180.0],
                colormap=["bwr"],
                colorbar_ticks=colorbar_ticks,
            )

        if "phi_sb" in dadict:
            colorbar_ticks = {
                "tickvals": np.linspace(0, 360, 5),
                "ticktext": np.linspace(0, 360, 5),
            }
            set_plot_option(
                dadict["phi_sb"],
                zlabel="phi_sb",
                zrange=[0.0, 360.0],
                colormap=["bwr"],
                colorbar_ticks=colorbar_ticks,
            )

        return dadict

    def analyze(self, ace, acb, dcb):
        t, f, result = self.svd(ace, acb, dcb)
        return self._setup_arrays(t, f, result)
