"""Verify real model execution without requiring the LecTrace viewer.
Only text/note/plot rendering is replaced by collectors; all tensor operations,
checkpoint loading and original-forward assertions execute unchanged.
"""
import ast
import json
import sys
from pathlib import Path
import platform
import numpy as np
import torch
from _runtime import HERE

def main():
    path=HERE/"01_eeg_tokenization.py"
    tree=ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=="lectrace")]
    outputs={"text":[],"note":[],"plot":[]}
    namespace={"__name__":"verification","__file__":str(path)}
    for kind in outputs:
        namespace[kind]=lambda value,k=kind: outputs[k].append(value)
    exec(compile(tree,str(path),"exec"),namespace)
    captured={}
    def tracer(frame,event,arg):
        if frame.f_code.co_filename==str(path) and frame.f_code.co_name=="main" and event=="return":
            captured.update(frame.f_locals)
        return tracer
    previous=sys.gettrace()
    try:
        sys.settrace(tracer)
        namespace["main"]()
    finally:
        sys.settrace(previous)
    expected={"input_shape":[2,4,512],"patch_shape":[2,4,7,128],"embedding_shape":[2,4,7,64],
              "temporal_input_shape":[8,7,64],"temporal_shape":[2,7,64],
              "spatial_input_shape":[14,4,64],"spatial_shape":[2,4,64],"fusion_shape":[2,12,64]}
    for k,v in expected.items(): assert captured[k]==v,(k,captured[k])
    assert len(outputs["plot"])==3
    for p in outputs["plot"]:
        assert p["data"]["values"]
        json.dumps(p,allow_nan=False)
    folds_path=HERE/"assets/folds.csv"
    import csv
    with folds_path.open(encoding="utf-8") as f: folds={r["sub"]:int(r["fold"]) for r in csv.DictReader(f)}
    for r in captured["meta"]["records"]:
        assert folds[r["sub"]]==captured["meta"]["checkpoint"]["fold"]==0
    assert captured["meta"]["results"]["bdi_delta_lo"]<0<captured["meta"]["results"]["bdi_delta_hi"]
    result={"status":"PASS: real inference and source parity; rendering not tested",
            "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
            "shapes":expected,"max_errors_vs_original_forward":captured["forward_errors"],
            "p_open":captured["probabilities"],"predicted_conditions":captured["predicted_conditions"],
            "true_conditions":["closed","open"],"plots_valid_json":len(outputs["plot"]),
            "n_text_blocks":len(outputs["text"]),"n_notes":len(outputs["note"]),
            "no_training":True,"LecTrace_browser_validation":"NOT RUN: package download blocked by DNS/network in authoring environment"}
    (HERE/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    # Save actual intermediate arrays for an optional scientific visual check.
    np.savez_compressed(HERE/"assets/verified_states.npz", x=captured["x"].numpy(), temporal=captured["temporal"].numpy(),
                        spatial=captured["spatial"].numpy(), probabilities=captured["p_open"].numpy())

if __name__=="__main__":
    main()

