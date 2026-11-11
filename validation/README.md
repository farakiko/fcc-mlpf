# validation — the data specification

This directory defines which EDM4hep collections a production must contain for the pipeline
to use it, and checks a file against that definition with one command:

```bash
python audit_edm4hep.py --detector <cld|idea|idea_o2> FILE.edm4hep.root [...]
```

The audit exits with code 0 and prints `VERDICT: PASS` when every required collection is
present. For each file it checks, in order:

1. that every required collection and MC-particle branch is stored (`--quick` stops here);
2. that every relation points to a collection stored in the file;
3. that the truth chain of the tracker works end to end (digitized hit → simulated hit →
   MC particle) for more than 95% of the tracker hits of the first five events;
4. for detectors with calorimeter truth, it reports the number of hit-to-truth links and the
   fraction of hits shared between several particles (informational, never a failure).

The audit should run on every delivery: our own HTCondor output, a colleague's dataset, the
first file of a new detector. The specification below is written with the CLD collection
names; the audit holds one collection map per detector (`CLD`, `IDEA`, `IDEA_O2` at the top
of `audit_edm4hep.py`) that fills the same roles with that detector's names. For the
simulated calorimeter hits it requires a minimum set (barrel collections and one
contributions collection), not the full list.

## A naming convention to know

podio stores the relation and vector members of a collection as separate branches whose
names start with an underscore: the hits of a track are in `_SiTracks_Refitted_trackerHits`
(with `.index` and `.collectionID`), its states in `_SiTracks_Refitted_trackStates`, the hits
of a cluster in `_PandoraClusters_hits`. Older files exposed some of these without the
underscore. A check written against the historical spelling reports collections as missing
that are present.

## Stage [T]: track finding

| collection | type | role |
|---|---|---|
| `VXDTrackerHits`, `VXDEndcapTrackerHits`, `ITrackerHits`, `ITrackerEndcapHits`, `OTrackerHits`, `OTrackerEndcapHits` | TrackerHitPlane | model input: digitized hits (position, energy deposit, du/dv, time), all six including the endcaps |
| `VXDTrackerHitRelations`, `VXDEndcapTrackerHitRelations`, `InnerTrackerBarrelHitsRelations`, `InnerTrackerEndcapHitsRelations`, `OuterTrackerBarrelHitsRelations`, `OuterTrackerEndcapHitsRelations` | relation, digitized → simulated | truth chain, first step |
| `VertexBarrelCollection`, `VertexEndcapCollection`, `InnerTrackerBarrelCollection`, `InnerTrackerEndcapCollection`, `OuterTrackerBarrelCollection`, `OuterTrackerEndcapCollection` | SimTrackerHit | truth chain, second step: simulated hit → MC particle, the training label |
| `SiTracks_Refitted` (with `_trackerHits`, `_trackStates`), `SiTracksMCTruthLink` | Track, link | classical baseline: hit sets, fitted states, track → MC particle |

The simulated tracker hits are the collections most at risk: productions before 2026-09
dropped them, which made those samples unusable for training.

## Stage [C]: calorimeter clustering

| collection | type | role |
|---|---|---|
| `ECALBarrel`, `ECALEndcap`, `HCALBarrel`, `HCALEndcap`, `HCALOther`, `MUON` | CalorimeterHit | model input: digitized calorimeter and muon-system hits |
| `CalohitMCTruthLink` | link with weight | truth: hit → MC particle. The weights split the calibrated hit energy among the contributing particles; 0.5% of the hits (1.5% of the energy) in tt̄ at 365 GeV are shared, which defines the fractional labels |
| `ECalBarrelCollection`, `ECalEndcapCollection`, `HCalBarrelCollection`, `HCalEndcapCollection`, `HCalRingCollection`, `YokeBarrelCollection`, `YokeEndcapCollection`, each with its `…Contributions` | SimCalorimeterHit | the MC energy decomposition per cell, below the level of the link weights |
| `PandoraClusters` (with `_PandoraClusters_hits`) | Cluster | classical baseline; the comparison is hit-based and needs the hit relation |

## Stage [PF]: particle flow

| collection | type | role |
|---|---|---|
| `MCParticles` with the full tree: `parents`, `daughters`, `vertex`, `endpoint`, `momentum`, `momentumAtEndpoint`, `generatorStatus`, `simulatorStatus`, `charge`, `mass`, `PDG` | MCParticle | targets: particles at the calorimeter entrance, primary and secondary particles, neutrinos for the missing momentum |
| `PandoraPFOs` (with its references), `RecoMCTruthLink` | ReconstructedParticle | classical baseline for the physics benchmarks (jets, missing momentum, efficiency and fake rate per class) |

[PF] also consumes everything required by [T] and [C].

## Bookkeeping

- `podio_metadata` must be intact: the audit resolves the `collectionID` of every relation
  through it, and every referenced collection must be stored.
- The production provenance is recorded next to the files: key4hep release, geometry,
  generator card and seed (the `.provenance.txt` files written by the production scripts,
  `production/README.md`).

## Other detectors

The specification is defined by roles: tracker digitized hits with their links to simulated
hits; calorimeter digitized hits with their truth links and the simulated hits with
contributions; the full MC particle tree; and the classical baselines where they exist. A new
detector satisfies it when each role is filled by its own collections. Drift-chamber and
dual-readout calorimeter hits fill the same roles with their own measurement content; the
canonical schema that holds them is described in
[`postprocessing/README.md`](../postprocessing/README.md).

- **IDEA o1** (`--detector idea`): the tracking-only digitized sample of A. De Vita
  (`production/idea/README.md`). It has no calorimeter and no reconstructed tracks; the
  audit skips those groups. The existing dataset passes, with every drift-chamber hit linked
  to its MC particle.
- **IDEA o2** (`--detector idea_o2`): tracker, muon system and dual-readout calorimeter
  (`production/idea_o2/README.md`). While the production omits the classical ECAL clustering
  and the direct calorimeter hit-to-particle links, the audit requires the digitized-to-
  simulated links and the simulated hits with contributions instead; both requirements are
  marked as temporary in `audit_edm4hep.py`.

## Known pitfalls

1. **Dropped simulated collections.** The default output slimming removes the truth chain,
   as happened in the productions before 2026-09. Keep `keep *`, or an explicit list that
   contains everything above.
2. **ddsim stops at initialization.** CLDConfig's `cld_steer.py` requires a tracking volume.
   CLD_o2_v08 (k4geo main) defines it through `<parallelworld_volume name="tracking_volume">`;
   older compact files (up to v06 in the released k4geo) define neither this nor the legacy
   `tracking_region_*` constants, and ddsim exits with code 1. With the handler active, the
   stored MC particles are pruned (about 600 per tt̄ event against about 820 without it), so
   samples with and without the handler must not be mixed.
3. **Version skew.** The 2026-04-08 stack ships an older CLDConfig (default geometry v07, no
   `--compactFile`) and a "v08" geometry directory with v07 content. We therefore vendor
   CLDConfig and the geometry (`production/cld/`) and pin the stack: the three are versioned
   independently on cvmfs.

## Slimming policy: produce everything, slim only derived files

Measured on a reference file (10 events, 34 MB compressed, 2026-10-01), 96% of the bytes are
needed by the specification. The simulated calorimeter contributions alone are 56%, and they
are the fractional clustering truth; calorimeter hits, simulated tracker hits and MC particles
follow. The high-level reconstruction we do not use (vertices, jets, duplicate track
collections, LumiCal, dQ/dx) is about 4%. Production therefore keeps `keep *`:

- the saving would be about 4%, against the cost of maintaining a keep-list per CLDConfig
  version;
- slimming at production time is irreversible, and it is what made the pre-2026-09 samples
  unusable;
- several unused collections have foreseeable uses (`BuildUpVertices_V0` for conversion
  finding, `dQdx` for particle identification, `SiTracksCT` for tracking comparisons);
- the right place to slim is the derived training cache, which is about 100 times smaller and
  can be rebuilt from the ROOT files at any time. If storage becomes the limit: produce the
  full files, build the caches, then archive or thin the ROOT files.
