#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

import h5py
import numpy as np
import pandas as pd

DIR_FMT = "%Y%m%d_%H%M%S"


def encode(x):
    return np.frombuffer(pickle.dumps(x), dtype=np.int8)


def decode(x):
    return pickle.loads(x.tobytes())


def save_hdf5(h5file, data, append=False):
    mode = "w" if append == False else "a"

    with h5py.File(h5file, mode) as fp:
        for ds in data:
            if hasattr(ds, "name"):  # DataArray object
                name = ds.name
                byte = encode(ds)
                fp.create_dataset(name, data=byte)
            elif "name" in ds:  # dict object
                name = ds["name"]
                byte = encode(ds)
                fp.create_dataset(name, data=byte)


def load_hdf5(h5file, tplot=None):
    data = list()
    with h5py.File(h5file, "r") as fp:
        for key in fp.keys():
            byte = fp.get(key)[()]
            data.append(decode(byte))

    # store tplot
    if tplot == True:
        store_tplot(data)

    return data


def store_tplot(data):
    import pytplot

    for ds in data:
        if hasattr(ds, "name"):  # DataArray object
            pytplot.data_quants[ds.name] = ds


def read_eventlist(filename):
    csv = pd.read_csv(filename, header=None, skiprows=1)
    tr1 = pd.to_datetime(csv.iloc[:, 0])
    tr2 = pd.to_datetime(csv.iloc[:, 1])
    return tr1, tr2


def doit(tr1, tr2, force=None):
    if len(tr1) != len(tr2):
        return

    log_omni = ""
    log_fast = ""
    log_brst = ""
    log_orbit = ""

    N = len(tr1)
    for i in range(N):
        # prepare directory
        dirname = tr1[i].strftime(DIR_FMT) + "-" + tr2[i].strftime(DIR_FMT)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if not os.path.isdir(dirname):
            print("ignoreing {} as it is not a directory".format(dirname))
            continue

        # load and save
        log_omni += load_and_save_omni(tr1[i], tr2[i], dirname, force)
        log_fast += load_and_save_fast(tr1[i], tr2[i], dirname, force)
        log_brst = load_and_save_brst(tr1[i], tr2[i], dirname, force)
        log_orbit += load_and_save_orbit(tr1[i], tr2[i], dirname, force)

    print("*** Log Message for omni ***")
    print(log_omni)

    print("*** Log Message for fast ***")
    print(log_fast)

    print("*** Log Message for brst ***")
    print(log_brst)

    print("*** Log Message for orbit ***")
    print(log_orbit)


def load_and_save_orbit(tr1, tr2, dirname, force=None):
    import pyspedas
    import pytplot

    fn = os.sep.join([dirname, "orbit.h5"])

    logmsg = ""

    # do not write
    if force != True and os.path.exists(fn):
        return logmsg

    # time range (+- 2 hours from the specified interval)
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = (pd.to_datetime(tr1) - np.timedelta64(2, "h")).strftime(fmt)
    t2 = (pd.to_datetime(tr2) + np.timedelta64(2, "h")).strftime(fmt)

    probe = [1, 2, 3, 4]
    kwargs = {
        "probe": [1, 2, 3, 4],
        "trange": [t1, t2],
        "time_clip": True,
    }

    try:
        pyspedas.mms.mec(data_rate="srvy", **kwargs)
    except Exception as e:
        logmsg += "Failed to load MEC for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    # save data
    varnames = list()
    suffix = [
        "mec_r_gse",
        "mec_v_gse",
    ]
    for i in range(4):
        sc = "mms%d_" % (i + 1)
        for s in suffix:
            varnames.append(sc + s)

    data = list()
    for name in varnames:
        if name in pytplot.data_quants:
            data.append(pytplot.data_quants[name])

    save_hdf5(fn, data)

    # clear
    pytplot.del_data()

    return logmsg


def load_and_save_fast(tr1, tr2, dirname, force=None):
    import pyspedas
    import pytplot

    fn = os.sep.join([dirname, "fast.h5"])

    logmsg = ""

    # do not write
    if force != True and os.path.exists(fn):
        return logmsg

    # time range (+- 10 min from the specified interval)
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = (pd.to_datetime(tr1) - np.timedelta64(10, "m")).strftime(fmt)
    t2 = (pd.to_datetime(tr2) + np.timedelta64(10, "m")).strftime(fmt)

    probe = [1, 2, 3, 4]
    kwargs = {
        "probe": [1, 2, 3, 4],
        "trange": [t1, t2],
        "time_clip": True,
    }

    ## FGM
    try:
        fmt = "*fgm_b_gse_srvy_l2"
        pyspedas.mms.fgm(data_rate="srvy", varformat=fmt, **kwargs)
    except Exception as e:
        logmsg += "Failed to load FGM for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    ## FPI moments
    try:
        fmt = "*(numberdensity|bulkv_gse|prestensor_gse|energyspectr_omni)*"
        pyspedas.mms.fpi(
            data_rate="fast", datatype=["des-moms", "dis-moms"], varformat=fmt, **kwargs
        )
    except Exception as e:
        logmsg += "Failed to load FPI for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    ## FEEPS
    try:
        pyspedas.mms.feeps(data_rate="srvy", datatype="electron", **kwargs)
    except Exception as e:
        logmsg += "Failed to load FEEPS for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    # save data
    varnames = list()
    suffix = [
        "fgm_b_gse_srvy_l2",
        "dis_numberdensity_fast",
        "des_numberdensity_fast",
        "dis_bulkv_gse_fast",
        "des_bulkv_gse_fast",
        "dis_temppara_fast",
        "des_temppara_fast",
        "dis_tempperp_fast",
        "des_tempperp_fast",
        "dis_prestensor_gse_fast",
        "des_prestensor_gse_fast",
        "dis_energyspectr_omni_fast",
        "des_energyspectr_omni_fast",
        "epd_feeps_srvy_l2_electron_intensity_omni",
    ]
    for i in range(4):
        sc = "mms%d_" % (i + 1)
        for s in suffix:
            varnames.append(sc + s)

    data = list()
    for name in varnames:
        if name in pytplot.data_quants:
            data.append(pytplot.data_quants[name])

    save_hdf5(fn, data)

    # clear
    pytplot.del_data()

    return logmsg


def load_and_save_brst(tr1, tr2, dirname, force=None):
    import pyspedas
    import pytplot

    fn = os.sep.join([dirname, "brst.h5"])

    logmsg = ""

    # do not write
    if force != True and os.path.exists(fn):
        return logmsg

    # time range (+- 1 min from the specified interval)
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = (pd.to_datetime(tr1) - np.timedelta64(1, "m")).strftime(fmt)
    t2 = (pd.to_datetime(tr2) + np.timedelta64(1, "m")).strftime(fmt)

    probe = [1, 2, 3, 4]
    kwargs = {
        "probe": [1, 2, 3, 4],
        "trange": [t1, t2],
        "time_clip": True,
    }

    ## FGM
    try:
        fmt = "*fgm_b_gse_brst_l2"
        pyspedas.mms.fgm(data_rate="brst", varformat=fmt, **kwargs)
    except Exception as e:
        logmsg += "Failed to load FGM for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    ## SCM
    try:
        fmt = "*scm_acb_gse_scb_brst_l2"
        pyspedas.mms.scm(data_rate="brst", varformat=fmt, **kwargs)
    except Exception as e:
        logmsg += "Failed to load SCM for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    ## EDP
    try:
        fmt = "*edp_dce_gse_brst_l2"
        pyspedas.mms.edp(data_rate="brst", varformat=fmt, **kwargs)
    except Exception as e:
        logmsg += "Failed to load EDP for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    ## FPI moments
    try:
        pyspedas.mms.fpi(data_rate="brst", **kwargs)
    except Exception as e:
        logmsg += "Failed to load FPI for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    ## FEEPS
    try:
        pyspedas.mms.feeps(data_rate="brst", datatype="electron", **kwargs)
        # FIXME: This does not work for some reasons
        # pyspedas.mms_feeps_pad(probe=probe, data_rate='brst')
    except Exception as e:
        logmsg += "Failed to load FEEPS for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    # save data
    varnames = list()
    suffix = [
        # FGM
        "fgm_b_gse_brst_l2",
        # SCM
        "scm_acb_gse_scb_brst_l2",
        # EDP
        "edp_dce_gse_brst_l2",
        # FPI-DES
        "des_errorflags_brst",
        "des_compressionloss_brst",
        "des_phi_brst",
        "des_phi_delta_brst",
        "des_dist_brst",
        "des_disterr_brst",
        "des_theta_brst",
        "des_theta_delta_brst",
        "des_energy_brst",
        "des_energy_delta_brst",
        "des_pitchangdist_lowen_brst",
        "des_pitchangdist_miden_brst",
        "des_pitchangdist_highen_brst",
        "des_energyspectr_par_brst",
        "des_energyspectr_anti_brst",
        "des_energyspectr_perp_brst",
        "des_energyspectr_omni_brst",
        "des_numberdensity_brst",
        "des_numberdensity_err_brst",
        "des_densityextrapolation_low_brst",
        "des_densityextrapolation_high_brst",
        "des_bulkv_gse_brst",
        "des_bulkv_err_brst",
        "des_prestensor_gse_brst",
        "des_prestensor_err_brst",
        "des_temppara_brst",
        "des_tempperp_brst",
        # FPI-DIS
        "dis_errorflags_brst",
        "dis_compressionloss_brst",
        "dis_phi_brst",
        "dis_phi_delta_brst",
        "dis_dist_brst",
        "dis_disterr_brst",
        "dis_theta_brst",
        "dis_theta_delta_brst",
        "dis_energy_brst",
        "dis_energy_delta_brst",
        "dis_pitchangdist_lowen_brst",
        "dis_pitchangdist_miden_brst",
        "dis_pitchangdist_highen_brst",
        "dis_energyspectr_par_brst",
        "dis_energyspectr_anti_brst",
        "dis_energyspectr_perp_brst",
        "dis_energyspectr_omni_brst",
        "dis_numberdensity_brst",
        "dis_numberdensity_err_brst",
        "dis_densityextrapolation_low_brst",
        "dis_densityextrapolation_high_brst",
        "dis_bulkv_gse_brst",
        "dis_bulkv_err_brst",
        "dis_prestensor_gse_brst",
        "dis_prestensor_err_brst",
        "dis_temppara_brst",
        "dis_tempperp_brst",
        # FEEPS
        "epd_feeps_brst_l2_electron_intensity_omni",
        "epd_feeps_brst_l2_electron_intensity_70-600keV_pad",
    ]
    for i in range(4):
        sc = "mms%d_" % (i + 1)
        for s in suffix:
            varnames.append(sc + s)

    data = list()
    for name in varnames:
        if name in pytplot.data_quants:
            data.append(pytplot.data_quants[name])

    save_hdf5(fn, data)

    # clear
    pytplot.del_data()

    return logmsg


def load_and_save_omni(tr1, tr2, dirname, force=None):
    import pyspedas
    import pytplot

    fn = os.sep.join([dirname, "omni.h5"])

    logmsg = ""

    # do not write
    if force != True and os.path.exists(fn):
        return logmsg

    # time range (+- 30 min from the specified interval)
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = (pd.to_datetime(tr1) - np.timedelta64(60, "m")).strftime(fmt)
    t2 = (pd.to_datetime(tr2) + np.timedelta64(60, "m")).strftime(fmt)

    # load
    try:
        pyspedas.omni.data(trange=[t1, t2], datatype="1min")
    except Exception as e:
        logmsg += "Failed to load OMNI for [{}, {}]".format(t1, t2)
        logmsg += " : " + str(e) + "\n"

    # save data
    mapnames = [
        ("proton_density", "omni_ni"),
        ("BX_GSE", "omni_bx"),
        ("BY_GSE", "omni_by"),
        ("BZ_GSE", "omni_bz"),
        ("Vx", "omni_vx"),
        ("Vy", "omni_vy"),
        ("Vz", "omni_vz"),
        ("flow_speed", "omni_vt"),
        ("Beta", "omni_beta"),
        ("Mach_num", "omni_mach"),
    ]
    data = list()
    for (old_name, new_name) in mapnames:
        if old_name in pytplot.data_quants:
            var = pytplot.data_quants[old_name]
            var.name = new_name
            data.append(var)

    save_hdf5(fn, data)

    # clear
    pytplot.del_data()

    return logmsg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Download Tool for MMS")
    parser.add_argument("target", nargs="+", type=str, help="event list files")
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="force download and overwrite existing files",
    )
    args = parser.parse_args()

    for target in args.target:
        if os.path.isfile(target):
            tr1, tr2 = read_eventlist(target)
            doit(tr1, tr2, force=args.force)
        else:
            print("Error: No such file {}".format(target))
