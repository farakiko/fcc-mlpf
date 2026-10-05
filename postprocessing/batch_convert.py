"""
Batch conversion driver: many EDM4hep ROOT files -> chunked canonical parquets, in parallel.

Chunks the sorted input list (files-per-chunk inputs -> one parquet), runs N workers,
skips chunks whose output already exists (resumable), audits every output, writes a
manifest. Keeps the sorted order so the file-level val convention ("first 10% of the
sorted list") carries over to the parquet chunks.

CLD example (anywhere with python+uproot+awkward+pyarrow):
  python batch_convert.py --adapter cld \
      --inputs "/eos/experiment/fcc/ee/simulation/key4hep_2026_04_08/365GeV/CLD_o2_v08/rec/ttbar_mlpf/*.edm4hep.root" \
      --outdir /eos/user/f/fmokhtar/fcc-mlpf/derived/cld_tt_parquet \
      --files-per-chunk 20 --nproc 8 --stack key4hep-2026-04-08

IDEA o1 (needs a key4hep env for podio; uses the digi->canonical wrapper per file,
then merges per chunk is NOT supported -- run with --files-per-chunk 1):
  python batch_convert.py --adapter idea --inputs ".../Zuds/digi/*.root" \
      --outdir ... --files-per-chunk 1 --nproc 4 --stack sw-nightlies-2026-09-13
"""
import argparse
import datetime
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def convert_chunk(adapter, files, out, stack, detector):
    if os.path.exists(out):
        return out, "skipped (exists)"
    tmp = out + ".tmp.parquet"
    try:
        if adapter == "cld":
            cmd = [sys.executable, os.path.join(HERE, "cld.py")] + files + [tmp, "--stack", stack]
            if detector:
                cmd += ["--detector", detector]
        elif adapter == "idea":
            assert len(files) == 1, "idea adapter converts one digi file per chunk"
            cmd = ["bash", os.path.join(HERE, "idea_digi_to_parquet.sh"), files[0], tmp, "--stack", stack]
        else:
            raise ValueError(adapter)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, "schema.py"), tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return out, f"AUDIT FAIL: {r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:200]}"
        os.rename(tmp, out)
        return out, "ok"
    except subprocess.CalledProcessError as e:
        return out, f"FAILED: {(e.stderr or e.stdout or '')[-300:]}"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, choices=["cld", "idea"])
    ap.add_argument("--inputs", required=True, help="glob of input ROOT files (quote it)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--files-per-chunk", type=int, default=20)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--stack", required=True, help="stack label for provenance metadata")
    ap.add_argument("--detector", default="", help="override detector label (cld adapter)")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    # condor striping: job N of M processes chunks with index % M == N (resumable; each
    # job writes its own MANIFEST_<N>.txt)
    ap.add_argument("--job-index", type=int, default=0)
    ap.add_argument("--num-jobs", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(args.inputs))
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        sys.exit(f"no inputs match {args.inputs}")
    os.makedirs(args.outdir, exist_ok=True)
    # SOURCE.txt: permanent record of where this derived dataset came from (written once,
    # idempotent across stripes) -- a required deliverable of every conversion.
    src_txt = os.path.join(args.outdir, "SOURCE.txt")
    if not os.path.exists(src_txt):
        with open(src_txt, "w") as f:
            f.write(f"derived dataset: canonical training parquet (postprocessing/README.md schema)\n"
                    f"source         : {args.inputs}\n"
                    f"source files   : {len(files)}{' (max-files='+str(args.max_files)+')' if args.max_files else ''}\n"
                    f"adapter        : postprocessing/{args.adapter}.py (fcc-mlpf)\n"
                    f"stack label    : {args.stack}\n"
                    f"files/chunk    : {args.files_per_chunk}\n"
                    f"created        : {datetime.date.today().isoformat()}\n"
                    f"note           : chunk order follows the SORTED source list; val convention =\n"
                    f"                 first 10% of chunks. Source data + per-file provenance +\n"
                    f"                 campaign ledger live alongside the source files.\n")
    chunks = [files[i:i + args.files_per_chunk] for i in range(0, len(files), args.files_per_chunk)]
    mine = [(ci, ch) for ci, ch in enumerate(chunks) if ci % args.num_jobs == args.job_index]
    print(f"{len(files)} files -> {len(chunks)} chunks ({args.files_per_chunk}/chunk); "
          f"this job: {len(mine)} chunks (stripe {args.job_index}/{args.num_jobs}), {args.nproc} workers")

    jobs, manifest = {}, []
    with ProcessPoolExecutor(max_workers=args.nproc) as ex:
        for ci, ch in mine:
            out = os.path.join(args.outdir, f"chunk_{ci:05d}.parquet")
            jobs[ex.submit(convert_chunk, args.adapter, ch, out, args.stack, args.detector)] = (ci, ch)
        done = fail = 0
        for fut in as_completed(jobs):
            ci, ch = jobs[fut]
            out, status = fut.result()
            manifest.append((ci, len(ch), ch[0], ch[-1], status))
            if status.startswith(("ok", "skipped")):
                done += 1
            else:
                fail += 1
                print(f"  [FAIL] chunk {ci}: {status}", flush=True)
            if (done + fail) % 25 == 0:
                print(f"  {done+fail}/{len(mine)} chunks ({fail} failed)", flush=True)

    mname = "MANIFEST.txt" if args.num_jobs == 1 else f"MANIFEST_{args.job_index:03d}.txt"
    with open(os.path.join(args.outdir, mname), "w") as f:
        f.write(f"# adapter={args.adapter} inputs={args.inputs} files_per_chunk={args.files_per_chunk} stack={args.stack}\n")
        for ci, n, first, last, status in sorted(manifest):
            f.write(f"chunk_{ci:05d}  {n:3d} files  {os.path.basename(first)} .. {os.path.basename(last)}  {status}\n")
    print(f"\n{done} ok / {fail} failed -> {args.outdir}  (MANIFEST.txt written)")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
