#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl

DIR_FMT = "%Y%m%d_%H%M%S"
JSON_FILENAME = "shockgeometry_mms{:1d}.json"


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


def check_consistency(dirname, quality):
    # check quality
    status = True and quality == 1
    if status == False:
        return status

    fn_js1 = os.sep.join([dirname, JSON_FILENAME.format(1)])
    fn_js2 = os.sep.join([dirname, JSON_FILENAME.format(2)])
    fn_js3 = os.sep.join([dirname, JSON_FILENAME.format(3)])
    fn_js4 = os.sep.join([dirname, JSON_FILENAME.format(4)])

    # check if all files exist
    fn_list = [fn_js1, fn_js2, fn_js3, fn_js4]
    status = status and np.alltrue(np.array([os.path.isfile(fn) for fn in fn_list]))
    if status == False:
        return status

    # check parameter consistency
    js = [0] * 4
    for i, fn in enumerate([fn_js1, fn_js2, fn_js3, fn_js4]):
        with open(fn, "r") as fp:
            js[i] = json.load(fp)

    for jsa, jsb in itertools.product(js, js):
        Ma_nif_i_val_a = jsa["Ma_nif_i"][0]
        Ma_nif_i_err_a = jsa["Ma_nif_i"][1]
        Ma_nif_i_val_b = jsb["Ma_nif_i"][0]
        Ma_nif_i_err_b = jsb["Ma_nif_i"][1]
        cos_tbn_val_a = jsa["cos_tbn"][0]
        cos_tbn_err_a = jsa["cos_tbn"][1]
        cos_tbn_val_b = jsb["cos_tbn"][0]
        cos_tbn_err_b = jsb["cos_tbn"][1]
        Ma_min = min(Ma_nif_i_val_a + Ma_nif_i_err_a, Ma_nif_i_val_b + Ma_nif_i_err_b)
        Ma_max = max(Ma_nif_i_val_a - Ma_nif_i_err_a, Ma_nif_i_val_b - Ma_nif_i_err_b)
        Tb_min = min(cos_tbn_val_a + cos_tbn_err_a, cos_tbn_val_b + cos_tbn_err_b)
        Tb_max = max(cos_tbn_val_a - cos_tbn_err_a, cos_tbn_val_b - cos_tbn_err_b)
        status = status and (Ma_min >= Ma_max) and (Tb_min >= Tb_max)

    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save parameters to CSV file.")
    parser.add_argument("target", nargs="+", type=str, help="target CSV files")
    parser.add_argument(
        "--sc",
        dest="sc",
        type=int,
        default=1,
        help="spacecraft number (1-4) [default: 1]",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default=None,
        required=True,
        help="output CSV filename",
    )
    args = parser.parse_args()

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
                dirname = t1.strftime(DIR_FMT) + "-" + t2.strftime(DIR_FMT)
                if check_consistency(dirname, q):
                    targetlist.append(dirname)
        else:
            print("Error: {} is not a file".format(target))

    # output to CSV file
    if args.output is not None:
        dictlist = []
        for dirname in targetlist:
            fn = os.sep.join([dirname, JSON_FILENAME]).format(args.sc)
            if os.path.isfile(fn):
                with open(fn, "r") as fp:
                    js = json.load(fp)
                    dictlist.append(json2dict(js, dirname))

        try:
            with open(args.output, "w") as fp:
                keys = list(dictlist[0].keys())
                fp.write("# " + ",".join(keys) + "\n")
                for d in dictlist:
                    fp.write("{}".format(d[keys[0]]))
                    for key in keys[1:]:
                        fp.write(",{}".format(d[key]))
                    fp.write("\n")
        except Exception as e:
            print(e)
