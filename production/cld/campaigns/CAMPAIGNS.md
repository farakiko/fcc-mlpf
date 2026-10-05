# CLD production campaign ledger

**Purpose: seed bookkeeping across campaigns.** Pythia/ddsim seeds share one space
(1 … 9×10⁸, Pythia's max). A reused seed = statistically identical events = silent dataset
duplication. Rules:
1. Every campaign's seeds are recorded here (file per campaign, extracted from the
   per-file `.provenance.txt` sidecars — the data itself is always the ground truth).
2. **Before submitting a new campaign**, pick `SEED_BASE` and verify disjointness:
   ```bash
   seq $SEED_BASE $((SEED_BASE + NJOBS - 1)) | sort > /tmp/new.txt
   sort -nu campaigns/*.seeds.txt | comm -12 - /tmp/new.txt   # MUST be empty
   ```
3. New campaigns use deterministic seeds (`SEED_BASE + process index`, see
   `gen_sim_rec.sh`) so a campaign occupies one contiguous, auditable block.
   (Campaign #1 predates this rule and used random seeds — hence the explicit list.)
4. After a campaign: extract and commit its seed list
   (`grep -h -o "seed=[0-9]*" <outdir>/*.provenance.txt | cut -d= -f2 | sort -n`)
   and drop a copy of it + this ledger next to the data on EOS.

| # | date | sample | jobs submitted / files delivered | events | seeds | output |
|---|---|---|---|---|---|---|
| 1 | 2026-10-01 | tt̄ @ 365 GeV, CLD_o2_v08, key4hep 2026-04-08 | 10,005 / 8,926 (≈1,080 jobs died, accepted) | 89,260 | **random**, list: `2026-10-01_ttbar365_100k.seeds.txt` (8,926 unique, verified no duplicates) | `/eos/experiment/fcc/ee/simulation/key4hep_2026_04_08/365GeV/CLD_o2_v08/rec/ttbar_mlpf/` (migrated from `/eos/user/f/fmokhtar/fcc-mlpf/data/cld_tt`) |
