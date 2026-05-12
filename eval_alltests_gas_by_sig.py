import os
import re
import json
import csv
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple


def run_cmd(cmd: List[str], cwd: str) -> Tuple[bool, str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return True, (r.stdout or '') + (r.stderr or '')
    except subprocess.CalledProcessError as e:
        out = (e.stdout or '') + (e.stderr or '')
        if not out:
            out = str(e)
        return False, out
    except FileNotFoundError as e:
        return False, f'FileNotFoundError: {e}'


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


def load_alltests_report(path: str) -> Dict[int, dict]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    details = obj.get('details') if isinstance(obj, dict) else None
    if not isinstance(details, list):
        return {}
    out: Dict[int, dict] = {}
    for r in details:
        if not isinstance(r, dict):
            continue
        tid = r.get('task_id')
        if tid is None:
            continue
        try:
            tid_int = int(tid)
        except Exception:
            continue
        out[tid_int] = r
    return out


def _extract_param_types_from_full_signature(full_signature: str) -> List[str]:
    if not full_signature:
        return []
    m = re.search(r"function\s+[A-Za-z0-9_]+\s*\((.*?)\)", full_signature, flags=re.DOTALL)
    if not m:
        return []
    raw = (m.group(1) or '').strip()
    if not raw:
        return []

    out: List[str] = []
    for seg in raw.split(','):
        seg = seg.strip()
        if not seg:
            continue
        tokens = [t for t in seg.split() if t]
        if len(tokens) <= 1:
            ptype = tokens[0] if tokens else ''
        else:
            # drop variable name
            ptype = ' '.join(tokens[:-1])
        ptype = re.sub(r"\s+", " ", ptype).strip()
        if ptype:
            out.append(ptype)
    return out


def _sig_type_variants(types: List[str]) -> List[str]:
    # Build candidate signature strings that may appear inside forge output, e.g. bytes1,bytes1
    # Also create variants without data location keywords to match outputs like (bytes,address)
    def _norm_no_space(s: str) -> str:
        return s.replace(' ', '')

    joined = ','.join(_norm_no_space(t) for t in types)
    variants = []
    if joined:
        variants.append(joined)

    # Remove memory/calldata/storage keywords
    cleaned: List[str] = []
    for t in types:
        t2 = re.sub(r"\b(memory|calldata|storage)\b", "", t)
        t2 = re.sub(r"\s+", " ", t2).strip()
        cleaned.append(_norm_no_space(t2))
    joined2 = ','.join(cleaned)
    if joined2 and joined2 not in variants:
        variants.append(joined2)

    return [v for v in variants if v]


_GAS_RE = re.compile(r"\(runs:\s*(\d+)\s*,\s*μ:\s*(\d+)\s*,\s*~:\s*(\d+)\)"
                     )


def extract_gas_for_signature(test_output: str, full_signature: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """Return (runs, mu, tilde, matched_line)."""
    if not test_output or not full_signature:
        return None, None, None, None

    types = _extract_param_types_from_full_signature(full_signature)
    variants = _sig_type_variants(types)
    if not variants:
        return None, None, None, None

    # Prefer line-based matching for stability.
    for line in test_output.splitlines():
        if '(runs:' not in line or 'μ:' not in line or '~:' not in line:
            continue
        line_nospace = line.replace(' ', '')
        if not any(f"({v})" in line_nospace for v in variants):
            continue
        m = _GAS_RE.search(line)
        if not m:
            continue
        try:
            runs = int(m.group(1))
            mu = int(m.group(2))
            tilde = int(m.group(3))
        except Exception:
            continue
        return runs, mu, tilde, line

    # Fallback: global scan for the first match that has the type signature nearby.
    for m in _GAS_RE.finditer(test_output):
        start = max(0, m.start() - 200)
        end = min(len(test_output), m.end() + 50)
        window = test_output[start:end]
        w_nospace = window.replace(' ', '')
        if any(f"({v})" in w_nospace for v in variants):
            try:
                return int(m.group(1)), int(m.group(2)), int(m.group(3)), window.strip().splitlines()[-1]
            except Exception:
                pass

    # Final fallback: sum ALL gas stats in this test output.
    # This is used when the output does not contain a per-test signature like testFoo(type1,type2).
    runs_sum = 0
    mu_sum = 0
    tilde_sum = 0
    n = 0
    for m in _GAS_RE.finditer(test_output):
        try:
            runs_sum += int(m.group(1))
            mu_sum += int(m.group(2))
            tilde_sum += int(m.group(3))
            n += 1
        except Exception:
            continue
    if n > 0:
        return runs_sum, mu_sum, tilde_sum, f'SUM_ALL_GAS_LINES count={n}'

    return None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, required=True, help='tasks json/jsonl (needs task_id + full_signature)')
    ap.add_argument('--output-dir', type=str, default='output')
    ap.add_argument('--alltests-report', type=str, required=True, help='eval_alltests_report.json to reuse pass_at_k + test_file')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--csv-out', type=str, default='eval_alltests_gas_by_sig_report.csv')
    ap.add_argument('--json-out', type=str, default='eval_alltests_gas_by_sig_report.json')
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    tindex = build_task_index(tasks)
    alltests_by_tid = load_alltests_report(args.alltests_report)

    rows: List[dict] = []
    mu_vals: List[int] = []
    tilde_vals: List[int] = []

    tids = sorted(alltests_by_tid.keys())
    for tid in tids:
        r = alltests_by_tid.get(tid) or {}
        if not isinstance(r, dict):
            continue
        # Only evaluate tasks that pass in eval_alltests
        if not bool(r.get('pass_at_k')):
            continue

        task_dir = os.path.join(args.output_dir, f'task_{tid}')
        if not os.path.isdir(task_dir):
            continue

        t = tindex.get(tid) or {}
        full_signature = str(t.get('full_signature', '')).strip() if isinstance(t, dict) else ''
        test_file = str(r.get('test_file', '')).strip()

        ok_test = False
        out = ''
        if test_file:
            ok_test, out = run_cmd(['forge', 'test', '--match-path', test_file], cwd=task_dir)
        else:
            ok_test, out = run_cmd(['forge', 'test'], cwd=task_dir)

        runs, mu, tilde, matched_line = extract_gas_for_signature(out, full_signature)

        if isinstance(mu, int):
            mu_vals.append(mu)
        if isinstance(tilde, int):
            tilde_vals.append(tilde)

        rows.append({
            'task_id': tid,
            'pass_at_k': int(bool(r.get('pass_at_k'))),
            'test_file': test_file,
            'forge_test_ok': int(bool(ok_test)),
            'gas_runs': runs if runs is not None else '',
            'gas_mu': mu if mu is not None else '',
            'gas_tilde': tilde if tilde is not None else '',
            'matched_gas_line': matched_line or '',
        })

    def _mean(xs: List[int]) -> Optional[float]:
        if not xs:
            return None
        return sum(xs) / len(xs)

    summary = {
        'tasks_considered_from_alltests': len(tids),
        'tasks_pass_at_k': len([1 for tid in tids if bool((alltests_by_tid.get(tid) or {}).get('pass_at_k'))]),
        'tasks_gas_extracted': len(mu_vals),
        'gas_mu_mean': _mean(mu_vals),
        'gas_tilde_mean': _mean(tilde_vals),
    }

    report = {'summary': summary, 'details': rows}
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['task_id', 'pass_at_k', 'test_file', 'forge_test_ok', 'gas_runs', 'gas_mu', 'gas_tilde', 'matched_gas_line'])
        for row in rows:
            w.writerow([
                row['task_id'],
                row['pass_at_k'],
                row['test_file'],
                row['forge_test_ok'],
                row['gas_runs'],
                row['gas_mu'],
                row['gas_tilde'],
                row['matched_gas_line'],
            ])

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
