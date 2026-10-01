# Data specification — EDM4hep collections required by the portable-MLPF pipeline

> **This is the authoritative list.** Every production (any detector) must satisfy it; every
> delivery is checked with `validation/audit_edm4hep.py` (which implements exactly this spec
> and exits non-zero on any gap). Collection names below are CLD's; for another detector the
> *roles* must map (see §5). Verified against the reference production of 2026-09-29
> (CLD_o2_v08, key4hep 2026-04-08; 495 files × 10 ttbar events).

## 0. Naming convention warning (learned the hard way)

podio encodes **relation/vector members as separate branches with a leading underscore**:
e.g. a track's hit list is `_SiTracks_Refitted_trackerHits` (with `.index`/`.collectionID`),
its states `_SiTracks_Refitted_trackStates`, a cluster's hits `_PandoraClusters_hits`.
Older files exposed some of these without the underscore. **Audits must match canonical
member names, not historical spellings** — we once mis-declared `trackStates` and
`PandoraClusters_hits` "missing" on exactly this.

## 1. Stage [T] — ML track finding (hits → tracks)

| collection | type | why |
|---|---|---|
| `VXDTrackerHits`, `VXDEndcapTrackerHits`, `ITrackerHits`, `ITrackerEndcapHits`, `OTrackerHits`, `OTrackerEndcapHits` | TrackerHitPlane | **model input**: digitized hits (position, eDep, du/dv, time, cov) — all six, incl. endcaps |
| `VXDTrackerHitRelations`, `VXDEndcapTrackerHitRelations`, `InnerTrackerBarrelHitsRelations`, `InnerTrackerEndcapHitsRelations`, `OuterTrackerBarrelHitsRelations`, `OuterTrackerEndcapHitsRelations` | LCRelation[digi→sim] | **truth chain, step 1**: digitized hit → sim hit |
| `VertexBarrelCollection`, `VertexEndcapCollection`, `InnerTrackerBarrelCollection`, `InnerTrackerEndcapCollection`, `OuterTrackerBarrelCollection`, `OuterTrackerEndcapCollection` | SimTrackerHit | **truth chain, step 2**: sim hit → `MCParticle` (the label). *These were dropped in pre-2026-09 productions — the single most important thing to keep.* |
| `SiTracks_Refitted` (+ `_trackerHits`, `_trackStates`), `SiTracksMCTruthLink` | Track, links | **baseline finder** for the head-to-head (hit sets, fitted states, track→MC) |

## 2. Stage [C] — ML calo clustering (hits → showers)

| collection | type | why |
|---|---|---|
| `ECALBarrel`, `ECALEndcap`, `HCALBarrel`, `HCALEndcap`, `HCALOther`, `MUON` | CalorimeterHit | **model input**: digitized calo + muon hits |
| `CalohitMCTruthLink` | link w/ weight | **truth**: hit → MCParticle; multi-link hits (~13%) with weights = the **fractional energy-sharing** labels |
| `ECalBarrelCollection`, `ECalEndcapCollection`, `HCalBarrelCollection`, `HCalEndcapCollection`, `HCalRingCollection`*, `YokeBarrelCollection`*, `YokeEndcapCollection`* (+ each `...Contributions`) | SimCalorimeterHit | **full MC energy decomposition** per cell (sub-hit contributions) — ground truth for fractional assignment beyond the link weights. *Names with * as present in the detector model.* |
| `PandoraClusters` (+ `_PandoraClusters_hits`) | Cluster | **baseline clustering** — hit-level double-majority comparison needs the hit relation |

## 3. Stage [PF] — particle-flow head (tracks + showers → particles)

| collection | type | why |
|---|---|---|
| `MCParticles` with **full tree**: `parents`/`daughters` refs, `vertex`, `endpoint`, `momentum`, `momentumAtEndpoint`, `generatorStatus`, `simulatorStatus`, `charge`, `mass`, `PDG` | MCParticle | targets: calo-entrance particle construction, primary/secondary split, status-1 truth, neutrinos for MET |
| `PandoraPFOs` (+ refs), `RecoMCTruthLink` | ReconstructedParticle | **baseline PF** for physics benchmarks (jets, MET, per-class eff/fake) |
| everything in §1 + §2 | | the PF head consumes both stages |

## 4. Bookkeeping (required)

- `podio_metadata` intact (collection-ID table — the audit resolves every relation's
  `collectionID` against it; **every referenced ID must be a stored collection**, no dangling).
- Production provenance recorded alongside the files: key4hep release, CLDConfig version/source,
  geometry compact file + its origin, generator card, seed scheme. (Our production scripts
  write this; see `docs/production_cld.md`.)

## 5. Porting to another detector (ALLEGRO / ILD / IDEA / CLD variants)

The spec is role-based: (tracker digi hits + digi→sim relations + sim hits), (calo digi hits +
hit→MC links + sim calo w/ contributions), (full MCParticle tree), (native baselines where they
exist). A new detector satisfies the spec when each role is filled by its corresponding
collections — names differ, the audit script takes a per-detector collection map
(see `validation/audit_edm4hep.py --detector`). Drift-chamber (IDEA) and dual-readout calo
hits fill the same roles with modality-specific payloads (see the mlpf repo's
`trackml_data_contract.md` for the downstream schema).

## 6. Known pitfalls (from the CLD forensics, 2026-09/10)

1. **Dropped sim collections**: default output-slimming kills the truth chain (pre-2026-09
   files). Keep `keep *` or an explicit list containing §1-§3.
2. **`Geant4TVUserParticleHandler` crash**: CLDConfig's `cld_steer.py` requires a tracking
   volume. CLD_o2_v08 (k4geo main) defines it via `<parallelworld_volume
   name="tracking_volume">`; older compacts (≤v06 in the released k4geo) define neither this
   nor the legacy `tracking_region_*` constants → **ddsim exits 1 at init**. Use v08+ (or
   patch constants in). NB: with the handler active, MCParticles are pruned (~600/ev vs ~820
   without) — a *physical* difference in stored secondaries; do not mix handler-on and
   handler-off samples in one training set.
3. **Release skew**: the 2026-04-08 stack ships an older CLDConfig (defaults to v07, no
   `--compactFile`) and a mislabeled "v08" geometry dir. Hence we **vendor** CLDConfig and the
   geometry in this repo (`production/cld/`) and pin the stack — the three versions are
   decoupled on cvmfs and must not be assumed consistent.

## 7. Slimming policy: produce FULL (`keep *`); slim only in derived caches

Decided 2026-10-01, on measurement (reference file, 10 events, 34 MB compressed):
**96% of the bytes are spec-needed** — sim-calo `*Contributions` alone are 56% (and they ARE
the fractional clustering truth), then calo hits, sim tracker hits, MCParticles. The genuinely
unused high-level reco (vertex/jet collections, `SiTracks`/`SiTracksCT` duplicates, LumiCal,
dQdx, …) totals **~4%**.

Therefore production stays at stock `keep *`:
- savings ≈ 4% vs the maintenance cost of a curated keep-list per CLDConfig version;
- production-time slimming is irreversible and is precisely what broke the pre-2026-09
  samples (pitfall #1) — this repo exists because of that mistake;
- several "unused" collections have anticipated uses (`BuildUpVertices_V0` → conversion
  finding for e/γ origin; `dQdx` → PID; `SiTracksCT` → tracking comparisons);
- the correct slimming layer is the **derived training cache** (ROOT → compact tensors,
  ~100× smaller, fully reproducible from ROOT). If storage ever binds at large event counts:
  produce full → build caches → archive/thin ROOT. Never slim at the source.
