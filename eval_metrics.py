import os
import re
import json
import csv
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_tasks(path: str) -> List[dict]:
    if not path:
        return []
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    items: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                continue
    return items


def build_task_index(tasks: List[dict]) -> Dict[int, dict]:
    idx: Dict[int, dict] = {}
    for t in tasks:
        tid = t.get('task_id')
        if tid is None:
            continue
        try:
            tid_int = int(tid)
        except Exception:
            continue
        idx[tid_int] = t
    return idx


def find_task_dirs(output_dir: str, task_id: int) -> List[str]:
    name = f'task_{task_id}'
    dirs: List[str] = []
    if not os.path.isdir(output_dir):
        return dirs
    p = os.path.join(output_dir, name)
    if os.path.isdir(p):
        dirs.append(p)
    return dirs


def load_key_map(key_path: str) -> Dict[str, str]:
    try:
        with open(key_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_forge_statuses(report: Optional[dict]) -> Tuple[Optional[bool], Optional[bool]]:
    if not report or not isinstance(report, dict):
        return None, None
    b = report.get('build', {}) if isinstance(report.get('build'), dict) else {}
    t = report.get('test', {}) if isinstance(report.get('test'), dict) else {}
    bs = b.get('status')
    ts = t.get('status')
    build_ok = (str(bs).upper() == 'OK') if bs is not None else None
    test_ok = (str(ts).upper() == 'OK') if ts is not None else None
    return build_ok, test_ok


def _extract_gas_from_forge_test_output(test_output: str) -> Dict[str, Optional[int]]:
    if not test_output:
        return {'gas_mu': None, 'gas_tilde': None}
    m = re.search(r"\(runs:\s*\d+\s*,\s*μ:\s*(\d+)\s*,\s*~:\s*(\d+)\)", test_output)
    if not m:
        return {'gas_mu': None, 'gas_tilde': None}
    try:
        return {'gas_mu': int(m.group(1)), 'gas_tilde': int(m.group(2))}
    except Exception:
        return {'gas_mu': None, 'gas_tilde': None}


def _load_token_usage_any(cand_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    direct = os.path.join(cand_dir, 'token_usage_fn_coding.json')
    if os.path.exists(direct):
        return _load_json(direct), direct
    for name in os.listdir(cand_dir):
        if not name.startswith('token_usage') or not name.endswith('.json'):
            continue
        p = os.path.join(cand_dir, name)
        if os.path.isfile(p):
            return _load_json(p), p
    return None, None


def _stage_stats_from_token_usage(tok: Optional[dict], stage: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not isinstance(tok, dict) or not stage:
        return None, None, None
    by_stage = tok.get('by_stage')
    if isinstance(by_stage, dict):
        st = by_stage.get(stage)
        if isinstance(st, dict):
            p = st.get('prompt')
            c = st.get('completion')
            calls = st.get('calls')
            if isinstance(p, int) and isinstance(c, int) and isinstance(calls, int):
                return p, c, calls

    raw = tok.get('raw_calls')
    if isinstance(raw, list):
        p_sum = 0
        c_sum = 0
        calls = 0
        for rc in raw:
            if not isinstance(rc, dict):
                continue
            if rc.get('stage') != stage:
                continue
            p = rc.get('prompt')
            c = rc.get('completion')
            if not isinstance(p, int) or not isinstance(c, int):
                continue
            p_sum += p
            c_sum += c
            calls += 1
        if calls > 0:
            return p_sum, c_sum, calls
    return None, None, None


def _load_security_any(cand_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    direct = os.path.join(cand_dir, 'slither_layered_report.json')
    if os.path.exists(direct):
        return _load_json(direct), direct
    return None, None


def _has_slither_after(cand_dir: str) -> bool:
    return os.path.exists(os.path.join(cand_dir, 'slither_after.json'))


def _has_slither_baseline(cand_dir: str) -> bool:
    return os.path.exists(os.path.join(cand_dir, 'slither_baseline.json'))


def _collect_high_medium_findings(report: Optional[dict]) -> List[dict]:
    if not report or not isinstance(report, dict):
        return []
    results = report.get('results')
    if not isinstance(results, dict):
        return []
    dets = results.get('detectors')
    if not isinstance(dets, list):
        return []
    out: List[dict] = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        impact = d.get('impact')
        if str(impact) not in ('High', 'Medium'):
            continue
        out.append(d)
    return out


def _finding_key(f: dict) -> str:
    if not isinstance(f, dict):
        return ''
    fid = f.get('id')
    if fid:
        return str(fid)
    return json.dumps({
        'check': f.get('check'),
        'impact': f.get('impact'),
        'confidence': f.get('confidence'),
        'first_markdown_element': f.get('first_markdown_element'),
    }, sort_keys=True, ensure_ascii=False)


def _finding_in_target_range(f: dict, target_basename: str, start: Optional[int], end: Optional[int]) -> bool:
    if not isinstance(f, dict):
        return False
    elements = f.get('elements')
    if not isinstance(elements, list):
        return False
    for el in elements:
        if not isinstance(el, dict):
            continue
        sm = el.get('source_mapping')
        if not isinstance(sm, dict):
            continue
        fnrel = sm.get('filename_relative')
        if not fnrel:
            continue
        fnrel_s = str(fnrel).replace('\\', '/').split('/')[-1]
        if target_basename and fnrel_s != target_basename:
            continue
        lines = sm.get('lines')
        if not isinstance(lines, list) or not lines:
            if start is None or end is None:
                return True
            continue
        try:
            lo = min(int(x) for x in lines if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit()))
            hi = max(int(x) for x in lines if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit()))
        except Exception:
            lo, hi = None, None
        if start is None or end is None or lo is None or hi is None:
            return True
        if not (hi < start or lo > end):
            return True
    return False


def _resolve_source_file_in_cand(cand_dir: str, source_id: str) -> Optional[str]:
    if not cand_dir or not source_id:
        return None
    try:
        sid = str(source_id).replace('\\', '/').strip()
        if not sid:
            return None
        rel = sid.lstrip('/')
        rel2 = rel[len('root/'):] if rel.startswith('root/') else rel

        candidates = [
            os.path.join(cand_dir, rel),
            os.path.join(cand_dir, rel2),
            os.path.join(cand_dir, 'root', rel2),
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                return p

        base = os.path.basename(rel)
        if base:
            for r, _, files in os.walk(cand_dir):
                for fn in files:
                    if fn == base:
                        pp = os.path.join(r, fn)
                        if os.path.isfile(pp):
                            return pp
    except Exception:
        return None
    return None


def _canonicalize_signature_head_from_text(text: str) -> str:
    if not text:
        return ''
    s = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.search(r"function\s+([A-Za-z0-9_]+)\s*\((.*?)\)", s, flags=re.DOTALL)
    if not m:
        return ''
    name = m.group(1)
    raw_params = (m.group(2) or '').strip()
    if not raw_params:
        return f"function {name}()"

    param_types: List[str] = []
    for seg in raw_params.split(','):
        seg = seg.strip()
        if not seg:
            continue
        tokens = [t for t in seg.split() if t]
        if len(tokens) <= 1:
            ptype = tokens[0] if tokens else ''
        else:
            ptype = ' '.join(tokens[:-1])
        ptype = re.sub(r"\s+", " ", ptype).strip().replace(' ', '')
        if ptype:
            param_types.append(ptype)
    return f"function {name}({','.join(param_types)})"


def _canonicalize_signature_head_from_full_signature(full_signature: str) -> str:
    if not full_signature:
        return ''
    m = re.search(r"(function\s+[A-Za-z0-9_]+\s*\(.*?\))", full_signature, flags=re.DOTALL)
    head = m.group(1) if m else full_signature
    return _canonicalize_signature_head_from_text(head)


def _locate_function_range_in_lines(
    lines: List[str],
    full_signature: str,
    expected_start: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    if not lines or not full_signature:
        return None, None

    target_head = _canonicalize_signature_head_from_full_signature(full_signature)
    if not target_head:
        return None, None
    mname = re.search(r"function\s+([A-Za-z0-9_]+)\s*\(", full_signature)
    func_name = mname.group(1) if mname else None
    if not func_name:
        return None, None

    cands: List[int] = []
    for i, ln in enumerate(lines, 1):
        if 'function' not in ln:
            continue
        if f"function {func_name}" not in ln and f"function\t{func_name}" not in ln:
            continue

        hdr = ''
        j = i - 1
        steps = 0
        while j < len(lines) and ')' not in hdr and steps < 60:
            hdr += lines[j].strip() + ' '
            if '{' in lines[j] or ';' in lines[j]:
                if ')' in hdr:
                    break
            j += 1
            steps += 1

        src_head = _canonicalize_signature_head_from_text(hdr)
        if src_head and src_head.replace(' ', '') == target_head.replace(' ', ''):
            cands.append(i)

    if not cands:
        return None, None
    start_line = min(cands, key=lambda x: abs(x - expected_start)) if expected_start is not None else cands[0]

    first_line = (lines[start_line - 1] or '').strip()
    if first_line.endswith(';'):
        return start_line, start_line

    brace_count = 0
    in_fn = False
    end_line: Optional[int] = None
    for idx0 in range(start_line - 1, len(lines)):
        s = lines[idx0]
        if not in_fn and '{' in s:
            in_fn = True
        if in_fn:
            brace_count += s.count('{') - s.count('}')
            if brace_count <= 0:
                end_line = idx0 + 1
                break
    return start_line, end_line


def _run_slither_in_dir(cand_dir: str, out_json_name: str) -> bool:
    out_path = os.path.join(cand_dir, out_json_name)
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass
    try:
        subprocess.run(
            ['slither', '.', '--compile-force-framework', 'foundry', '--json', out_json_name],
            cwd=cand_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return os.path.exists(out_path)


def _security_pass_from_layered(layered: Optional[dict]) -> Optional[bool]:
    if layered is None:
        return None
    if not isinstance(layered, dict):
        return None
    st = layered.get('status')
    if st is None:
        return None
    return str(st).upper() == 'OK'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, required=True)
    ap.add_argument('--output-dir', type=str, default='output')
    ap.add_argument('--key-path', type=str, default=os.path.join(os.getcwd(), 'key.json'))
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--recompute-slither', action='store_true')
    ap.add_argument('--after-json-name', type=str, default='slither_after_eval.json')
    ap.add_argument('--csv-out', type=str, default='eval_metrics_report.csv')
    ap.add_argument('--json-out', type=str, default='eval_metrics_report.json')
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    tindex = build_task_index(tasks)
    key_map = load_key_map(args.key_path)

    def resolve_mapping(source_id: str) -> Optional[str]:
        if not source_id:
            return None
        if source_id in key_map:
            return key_map[source_id]
        base = os.path.basename(source_id.replace('\\', '/'))
        for k in key_map.keys():
            if os.path.basename(k.replace('\\', '/')) == base:
                return key_map[k]
        return None

    task_ids: List[int] = []
    if os.path.isdir(args.output_dir):
        for name in os.listdir(args.output_dir):
            if name.startswith('task_') and os.path.isdir(os.path.join(args.output_dir, name)):
                tid = name[len('task_'):]
                if tid.isdigit():
                    task_ids.append(int(tid))
    task_ids.sort()

    details = []

    security_pass_at_k = 0
    gas_mu_values: List[int] = []
    gas_tilde_values: List[int] = []
    token_prompt_values: List[int] = []
    token_completion_values: List[int] = []
    token_total_values: List[int] = []

    token_prompt_sum = 0
    token_completion_sum = 0
    token_total_sum = 0
    token_prompt_count = 0
    token_completion_count = 0
    token_total_count = 0

    fn_coding_prompt_sum = 0
    fn_coding_completion_sum = 0
    fn_coding_calls_sum = 0
    fn_secure_refine_prompt_sum = 0
    fn_secure_refine_completion_sum = 0
    fn_secure_refine_calls_sum = 0

    for tid in task_ids:
        t = tindex.get(tid, {})
        source_id = str(t.get('source_id', '')).strip() if isinstance(t, dict) else ''
        _ = resolve_mapping(source_id)

        full_signature = str(t.get('full_signature', '')).strip() if isinstance(t, dict) else ''

        target_basename = os.path.basename(source_id.replace('\\', '/')) if source_id else ''
        start = None
        end = None
        if isinstance(t, dict):
            try:
                start = int(t.get('start')) if t.get('start') is not None else None
            except Exception:
                start = None
            try:
                end = int(t.get('end')) if t.get('end') is not None else None
            except Exception:
                end = None

        cand_dirs = find_task_dirs(args.output_dir, tid)
        k = max(1, min(args.k, len(cand_dirs) if cand_dirs else 1))

        cand_security: List[Optional[bool]] = []
        cand_gas: List[Dict[str, Optional[int]]] = []
        cand_tokens: List[Dict[str, Optional[int]]] = []

        for d in cand_dirs:
            # 1. 优先读取 Forge 状态
            forge_report = _load_json(os.path.join(d, 'forge_test_report.json'))
            build_ok, test_ok = _parse_forge_statuses(forge_report)
            
            # 2. 计算安全性逻辑 (只有 test_ok 为 True 时才可能为 True)
            if test_ok is not True:
                cand_security.append(False)
            elif args.recompute_slither:
                if not _has_slither_baseline(d):
                    cand_security.append(False)
                else:
                    ok = _run_slither_in_dir(d, args.after_json_name)
                    if not ok:
                        cand_security.append(False)
                    else:
                        resolved_source = _resolve_source_file_in_cand(d, source_id)
                        dyn_start, dyn_end = start, end
                        dyn_basename = target_basename
                        if resolved_source and full_signature:
                            try:
                                with open(resolved_source, 'r', encoding='utf-8') as fsrc:
                                    src_lines = fsrc.readlines()
                                s2, e2 = _locate_function_range_in_lines(src_lines, full_signature, expected_start=start)
                                if s2 is not None and e2 is not None:
                                    dyn_start, dyn_end = s2, e2
                                    dyn_basename = os.path.basename(resolved_source.replace('\\', '/'))
                            except Exception:
                                pass
                        baseline_report = _load_json(os.path.join(d, 'slither_baseline.json'))
                        after_report = _load_json(os.path.join(d, args.after_json_name))
                        baseline_findings = [
                            f for f in _collect_high_medium_findings(baseline_report)
                            if _finding_in_target_range(f, dyn_basename, dyn_start, dyn_end)
                        ]
                        after_findings = [
                            f for f in _collect_high_medium_findings(after_report)
                            if _finding_in_target_range(f, dyn_basename, dyn_start, dyn_end)
                        ]
                        baseline_keys = {_finding_key(f) for f in baseline_findings}
                        new_findings = [f for f in after_findings if _finding_key(f) not in baseline_keys]
                        cand_security.append(len(new_findings) == 0)
            else:
                layered, _layered_path = _load_security_any(d)
                if _has_slither_after(d):
                    cand_security.append(_security_pass_from_layered(layered))
                else:
                    cand_security.append(False)

            # 3. 计算 Gas 和 Token 逻辑
            gas = {'gas_mu': None, 'gas_tilde': None}
            if isinstance(forge_report, dict):
                test = forge_report.get('test')
                if isinstance(test, dict):
                    gas = _extract_gas_from_forge_test_output(str(test.get('output', '')))
            cand_gas.append(gas)

            tok, _tok_path = _load_token_usage_any(d)
            rec = {'prompt': None, 'completion': None, 'tokens': None}
            if isinstance(tok, dict):
                total = tok.get('total')
                if isinstance(total, dict):
                    rec['prompt'] = total.get('prompt')
                    rec['completion'] = total.get('completion')
                    rec['tokens'] = total.get('tokens')

                p, c, calls = _stage_stats_from_token_usage(tok, 'fn_coding')
                if isinstance(p, int) and isinstance(c, int) and isinstance(calls, int):
                    fn_coding_prompt_sum += p
                    fn_coding_completion_sum += c
                    fn_coding_calls_sum += calls
                p, c, calls = _stage_stats_from_token_usage(tok, 'fn_secure_refine')
                if isinstance(p, int) and isinstance(c, int) and isinstance(calls, int):
                    fn_secure_refine_prompt_sum += p
                    fn_secure_refine_completion_sum += c
                    fn_secure_refine_calls_sum += calls
            cand_tokens.append(rec)

        sec_ok = any(v is True for v in cand_security[:k]) if cand_security else False
        if sec_ok:
            security_pass_at_k += 1

        def _first_int(values: List[Optional[int]]) -> Optional[int]:
            for v in values:
                if isinstance(v, int):
                    return v
            return None

        first_mu = _first_int([g.get('gas_mu') for g in cand_gas[:k]])
        first_tilde = _first_int([g.get('gas_tilde') for g in cand_gas[:k]])
        first_prompt = _first_int([tt.get('prompt') for tt in cand_tokens[:k]])
        first_completion = _first_int([tt.get('completion') for tt in cand_tokens[:k]])
        first_total_tokens = _first_int([tt.get('tokens') for tt in cand_tokens[:k]])

        if isinstance(first_mu, int):
            gas_mu_values.append(first_mu)
        if isinstance(first_tilde, int):
            gas_tilde_values.append(first_tilde)
        if isinstance(first_prompt, int):
            token_prompt_values.append(first_prompt)
            token_prompt_sum += first_prompt
            token_prompt_count += 1
        if isinstance(first_completion, int):
            token_completion_values.append(first_completion)
            token_completion_sum += first_completion
            token_completion_count += 1
        if isinstance(first_total_tokens, int):
            token_total_values.append(first_total_tokens)
            token_total_sum += first_total_tokens
            token_total_count += 1

        details.append({
            'task_id': tid,
            'num_candidates': len(cand_dirs),
            'security_candidates': cand_security,
            'security_pass_at_k': sec_ok,
            'gas_mu_first': first_mu,
            'gas_tilde_first': first_tilde,
            'token_prompt_first': first_prompt,
            'token_completion_first': first_completion,
            'token_total_first': first_total_tokens,
        })

    total_tasks = len(details)

    def _mean(xs: List[int]) -> Optional[float]:
        if not xs:
            return None
        return sum(xs) / len(xs)

    summary = {
        'tasks': total_tasks,
        'k': args.k,
        'security_pass_at_k_tasks': security_pass_at_k,
        'security_pass_at_k_rate': (security_pass_at_k / total_tasks) if total_tasks else 0.0,
        'gas_mu_mean': _mean(gas_mu_values),
        'gas_tilde_mean': _mean(gas_tilde_values),
        'token_prompt_mean': _mean(token_prompt_values),
        'token_completion_mean': _mean(token_completion_values),
        'token_total_mean': _mean(token_total_values),
        'token_prompt_sum': token_prompt_sum,
        'token_completion_sum': token_completion_sum,
        'token_total_sum': token_total_sum,
        'token_prompt_count': token_prompt_count,
        'token_completion_count': token_completion_count,
        'token_total_count': token_total_count,
        'fn_coding_prompt_sum': fn_coding_prompt_sum,
        'fn_coding_completion_sum': fn_coding_completion_sum,
        'fn_coding_calls_sum': fn_coding_calls_sum,
        'fn_coding_prompt_avg_per_call': (fn_coding_prompt_sum / fn_coding_calls_sum) if fn_coding_calls_sum else None,
        'fn_coding_completion_avg_per_call': (fn_coding_completion_sum / fn_coding_calls_sum) if fn_coding_calls_sum else None,
        'fn_secure_refine_prompt_sum': fn_secure_refine_prompt_sum,
        'fn_secure_refine_completion_sum': fn_secure_refine_completion_sum,
        'fn_secure_refine_calls_sum': fn_secure_refine_calls_sum,
        'fn_secure_refine_prompt_avg_per_call': (fn_secure_refine_prompt_sum / fn_secure_refine_calls_sum) if fn_secure_refine_calls_sum else None,
        'fn_secure_refine_completion_avg_per_call': (fn_secure_refine_completion_sum / fn_secure_refine_calls_sum) if fn_secure_refine_calls_sum else None,
    }

    report = {'summary': summary, 'details': details}
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'task_id',
            'num_candidates',
            'security_pass_at_k',
            'gas_mu_first',
            'gas_tilde_first',
            'token_prompt_first',
            'token_completion_first',
            'token_total_first',
        ])
        for r in details:
            w.writerow([
                r['task_id'],
                r['num_candidates'],
                int(bool(r['security_pass_at_k'])),
                r['gas_mu_first'] if r['gas_mu_first'] is not None else '',
                r['gas_tilde_first'] if r['gas_tilde_first'] is not None else '',
                r['token_prompt_first'] if r['token_prompt_first'] is not None else '',
                r['token_completion_first'] if r['token_completion_first'] is not None else '',
                r['token_total_first'] if r['token_total_first'] is not None else '',
            ])

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
