import os
import json
import argparse
import subprocess
import glob
import csv
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
        raise ValueError('--tasks is required for this evaluator (needs source_id per task)')
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
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
                items.append(o)
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
    with open(key_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _parse_public_pass_from_report_json(report: Optional[dict]) -> Optional[bool]:
    if not report or not isinstance(report, dict):
        return None
    test = report.get('test')
    if not isinstance(test, dict):
        return None
    st = test.get('status')
    if st is None:
        return None
    return str(st).upper() == 'OK'


def _parse_public_pass_from_report_txt(text: str) -> Optional[bool]:
    if not text:
        return None
    s = text.upper()
    if 'STATUS' in s and 'TEST' in s:
        if 'OK' in s and 'FAIL' not in s and 'ERROR' not in s:
            return True
    if '[FAIL' in s or 'FAILED' in s:
        return False
    if '[PASS' in s and '[FAIL' not in s:
        return True
    return None


def load_public_pass_from_candidate_dir(cand_dir: str) -> Optional[bool]:
    rj = os.path.join(cand_dir, 'forge_test_report.json')
    if os.path.exists(rj):
        try:
            with open(rj, 'r', encoding='utf-8') as f:
                report = json.load(f)
            v = _parse_public_pass_from_report_json(report)
            if v is not None:
                return v
        except Exception:
            pass

    rt = os.path.join(cand_dir, 'forge_test_report.txt')
    if os.path.exists(rt):
        try:
            with open(rt, 'r', encoding='utf-8') as f:
                text = f.read()
            return _parse_public_pass_from_report_txt(text)
        except Exception:
            return None
    return None


def _find_test_file(dest_root: str, mapping_val: str) -> Optional[str]:
    """Locate the test file inside the copied Foundry workspace.
    Strategy:
      1) Search by basename under dest_root/test/**
      2) Fallback: if mapping contains '/test/', use the subpath starting at 'test/'
    """
    try:
        base = os.path.basename(mapping_val.replace('\\', '/'))
        test_root = os.path.join(dest_root, 'test')
        if os.path.isdir(test_root):
            for p, _, files in os.walk(test_root):
                for f in files:
                    if f == base and f.endswith('.sol'):
                        return os.path.join(p, f)
        v = mapping_val.replace('\\', '/').lstrip('/')
        idx = v.find('/test/')
        if idx != -1:
            sub = v[idx + 1:]  # keep 'test/...'
            cand = os.path.join(dest_root, sub)
            if os.path.exists(cand):
                return cand
    except Exception:
        return None
    return None


def eval_candidate_with_testfile(cand_dir: str, test_rel_path: str) -> Tuple[bool, bool, str, str]:
    """Run forge build and then run tests for the given test file only.
    Returns: (ok_build, ok_test, build_out, test_out)
    """
    ok_build, build_out = run_cmd(['forge', 'build'], cwd=cand_dir)
    # run all tests inside the mapped test file
    ok_test, test_out = run_cmd(['forge', 'test', '--match-path', test_rel_path], cwd=cand_dir)
    return ok_build, ok_test, build_out, test_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, required=True, help='path to tasks json/jsonl; must include task_id and source_id')
    ap.add_argument('--output-dir', type=str, default='output')
    ap.add_argument('--key-path', type=str, default=os.path.join(os.getcwd(), 'projects', 'sol_fn_miniloop', 'key.json'))
    ap.add_argument('--k', type=int, default=1, help='compute compile@k and pass@k over first k candidates for each task')
    ap.add_argument('--csv-out', type=str, default='eval_alltests_report.csv')
    ap.add_argument('--json-out', type=str, default='eval_alltests_report.json')
    args = ap.parse_args()

    # load inputs
    tasks = load_tasks(args.tasks)
    tindex = build_task_index(tasks)
    key_map = load_key_map(args.key_path)

    def resolve_mapping(source_id: str) -> Optional[str]:
        if not source_id:
            return None
        if source_id in key_map:
            return key_map[source_id]
        # fuzzy match by basename
        base = os.path.basename(source_id.replace('\\', '/'))
        for k in key_map.keys():
            if os.path.basename(k.replace('\\', '/')) == base:
                return key_map[k]
        return None

    rows = []
    compile_at_k_count = 0
    pass_at_k_count = 0
    public_pass_at_k_count = 0

    # Decide tasks to evaluate: only those that have task_* folders present
    task_ids = []
    if os.path.isdir(args.output_dir):
        for name in os.listdir(args.output_dir):
            if name.startswith('task_') and os.path.isdir(os.path.join(args.output_dir, name)):
                tid = name[len('task_'):]
                if tid.isdigit():
                    task_ids.append(int(tid))
    task_ids.sort()

    for tid in task_ids:
        t = tindex.get(tid)
        if not t:
            # Skip tasks that have no task metadata (no source_id)
            rows.append({
                'task_id': tid,
                'num_candidates': 0,
                'compile_candidates': [],
                'pass_candidates': [],
                'compile_at_k': False,
                'pass_at_k': False,
                'test_file': '',
                'mapping_found': False,
            })
            continue

        source_id = str(t.get('source_id', '')).strip()
        mapping_val = resolve_mapping(source_id)
        cand_dirs = find_task_dirs(args.output_dir, tid)
        cand_build: List[bool] = []
        cand_pass: List[bool] = []
        cand_public_pass: List[Optional[bool]] = []
        test_rel_for_rows = ''

        for d in cand_dirs:
            cand_public_pass.append(load_public_pass_from_candidate_dir(d))
            # locate test file inside this candidate workspace
            test_abs = _find_test_file(d, mapping_val) if mapping_val else None
            if not test_abs:
                cand_build.append(False)
                cand_pass.append(False)
                continue
            test_rel = os.path.relpath(test_abs, start=d).replace('\\', '/')
            test_rel_for_rows = test_rel
            okb, okt, _, _ = eval_candidate_with_testfile(d, test_rel)
            cand_build.append(okb)
            cand_pass.append(okt)

        k = max(1, min(args.k, len(cand_build) if cand_build else 1))
        cb = any(cand_build[:k]) if cand_build else False
        pb = any(cand_pass[:k]) if cand_pass else False
        pubb = any(v is True for v in cand_public_pass[:k]) if cand_public_pass else False
        if cb:
            compile_at_k_count += 1
        if pb:
            pass_at_k_count += 1
        if pubb:
            public_pass_at_k_count += 1

        rows.append({
            'task_id': tid,
            'num_candidates': len(cand_dirs),
            'compile_candidates': cand_build,
            'pass_candidates': cand_pass,
            'public_pass_candidates': cand_public_pass,
            'compile_at_k': cb,
            'pass_at_k': pb,
            'public_pass_at_k': pubb,
            'test_file': test_rel_for_rows,
            'mapping_found': mapping_val is not None,
        })

    total = len(rows)
    summary = {
        'tasks': total,
        'k': args.k,
        'compile_at_k_tasks': compile_at_k_count,
        'pass_at_k_tasks': pass_at_k_count,
        'public_pass_at_k_tasks': public_pass_at_k_count,
        'compile_at_k_rate': (compile_at_k_count / total) if total else 0.0,
        'pass_at_k_rate': (pass_at_k_count / total) if total else 0.0,
        'public_pass_at_k_rate': (public_pass_at_k_count / total) if total else 0.0
    }

    report = {'summary': summary, 'details': rows}
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['task_id', 'num_candidates', 'compile_at_k', 'pass_at_k', 'public_pass_at_k'])
        for r in rows:
            w.writerow([
                r['task_id'],
                r['num_candidates'],
                int(r['compile_at_k']),
                int(r['pass_at_k']),
                int(r['public_pass_at_k']),
            ])

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
