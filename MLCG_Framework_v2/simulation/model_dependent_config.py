#!/usr/bin/env python3
"""Load and provenance-track model-dependent workflow configuration.

Generic framework code contains no model-specific values here. Workflow wrappers
select named sections from an external JSON file. Explicit environment overrides
remain possible and are recorded by the provenance writer.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shlex
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "mlcg_model_dependent_workflow_config"

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def load_config(path: Path) -> dict[str, Any]:
    path=path.expanduser().resolve()
    data=json.loads(path.read_text())
    if data.get('schema_version') != SCHEMA_VERSION: raise ValueError('Unsupported model-dependent config schema_version')
    if data.get('kind') != KIND: raise ValueError('Unsupported model-dependent config kind')
    if not isinstance(data.get('sections'), dict) or not data['sections']: raise ValueError("Missing non-empty 'sections'")
    return data

def _shell_value(v: Any) -> str:
    if isinstance(v,bool): return '1' if v else '0'
    if isinstance(v,(str,int,float)): return str(v)
    if isinstance(v,list):
        if any(isinstance(x,(list,dict)) for x in v): return json.dumps(v,separators=(',',':'),sort_keys=True)
        return ' '.join(str(x) for x in v)
    if isinstance(v,dict): return json.dumps(v,separators=(',',':'),sort_keys=True)
    if v is None: return ''
    raise TypeError(type(v).__name__)

def merge_sections(data: dict[str, Any], names: list[str]) -> dict[str, Any]:
    out={}
    for name in names:
        if name not in data['sections']: raise KeyError(f'Missing model-dependent configuration section: {name}')
        sec=data['sections'][name]
        if not isinstance(sec,dict): raise ValueError(f'Section {name!r} must be an object')
        for k,v in sec.items():
            if not k or not k.replace('_','A').isalnum() or not k[0].isalpha(): raise ValueError(f'Unsafe configuration key: {k!r}')
            out[k]=v
    return out

def resolved_values(data, names, *, preserve_env):
    vals={}; sources={}
    for k,v in merge_sections(data,names).items():
        cv=_shell_value(v)
        if preserve_env and k in os.environ:
            vals[k]=os.environ[k]; sources[k]='environment_override'
        else:
            vals[k]=cv; sources[k]='model_config'
    return vals,sources

def cmd_export(args):
    path=args.config.expanduser().resolve(); data=load_config(path)
    vals,_=resolved_values(data,args.sections,preserve_env=args.preserve_env)
    vals.update(MODEL_DEPENDENT_CONFIG_PATH=str(path),MODEL_DEPENDENT_CONFIG_SHA256=sha256_file(path),MODEL_DEPENDENT_CONFIG_SECTIONS=' '.join(args.sections))
    for k,v in vals.items(): print(f'export {k}={shlex.quote(v)}')

def cmd_provenance(args):
    path=args.config.expanduser().resolve(); data=load_config(path); configured=merge_sections(data,args.sections)
    ct={k:_shell_value(v) for k,v in configured.items()}; resolved={}; sources={}
    for k,v in ct.items():
        resolved[k]=os.environ.get(k,v)
        sources[k]='environment_override' if k in os.environ and os.environ[k] != v else 'model_config'
    report={'schema_version':1,'kind':'model_dependent_workflow_config_provenance','model_id':data.get('model_id'),
            'config':str(path),'config_sha256':sha256_file(path),'sections':args.sections,
            'configured_values':ct,'resolved_values':resolved,'value_sources':sources,
            'calibration_provenance':data.get('calibration_provenance',{})}
    out=args.output.expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(f'[CONFIG] provenance: {out}')

def cmd_validate(args):
    path=args.config.expanduser().resolve(); data=load_config(path)
    for s in args.sections: merge_sections(data,[s])
    print(f'[PASS] model-dependent config: {path}')
    print(f'[PASS] SHA256: {sha256_file(path)}')
    if args.sections: print(f"[PASS] sections: {' '.join(args.sections)}")

def main():
    p=argparse.ArgumentParser(description=__doc__); sp=p.add_subparsers(dest='cmd',required=True)
    x=sp.add_parser('export-shell'); x.add_argument('--config',required=True,type=Path); x.add_argument('--sections',nargs='+',required=True); x.add_argument('--preserve-env',action='store_true'); x.set_defaults(func=cmd_export)
    x=sp.add_parser('provenance'); x.add_argument('--config',required=True,type=Path); x.add_argument('--sections',nargs='+',required=True); x.add_argument('--output',required=True,type=Path); x.set_defaults(func=cmd_provenance)
    x=sp.add_parser('validate'); x.add_argument('--config',required=True,type=Path); x.add_argument('--sections',nargs='*',default=[]); x.set_defaults(func=cmd_validate)
    a=p.parse_args(); a.func(a)
if __name__=='__main__': main()
