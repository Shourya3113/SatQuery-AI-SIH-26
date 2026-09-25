# Geospatial Band & Basemap Contract

**Owner:** Misha (Database & Geospatial Pipeline Lead)
**Branch:** `feature/misha-geospatial-v2`
**Status:** Uncommitted on top of `0cc7f38` — needs review from Peter (Git Master) before merging to `develop`.
**Audience:** Everyone whose module reads raster bands, renders a map, or cites per-band metrics.

---

## 1. Why this document exists

This branch changes **how a raster's bands are identified**. That resolution is not a private
detail of `services/geospatial.py` — half the codebase either reads bands by hard-coded position
or displays per-band results. Anything that assumes "channel 3 is NIR" is now relying on an
explicit, documented contract instead of an implicit convention.

Read §4 to find your module. Read §5 for the change your branch needs.

---

## 2. What this branch changes

| File | Change | Kind |
| --- | --- | --- |
| `services/geospatial.py` | Positional band fallback for unlabelled rasters in `raster_index()` | Behaviour change |
| `tests/test_geospatial.py` | 2 new tests locking the fallback and its boundary | Test |
| `scripts/dev_workbench.py` | Basemap tiles: OpenStreetMap → keyless Esri layers | Dev tooling |

Nothing else in the pipeline changes shape. No signature was altered, no schema renamed.

---

## 3. The band resolution contract

### 3.1 Resolution precedence (`services/geospatial.py`)

`raster_index(path, index)` resolves each **role** a spectral index needs, in this order:

1. **Band descriptions (authoritative).** `_resolve_band()` (line 468) matches `_BAND_HINTS`
   (line 418) against the raster's band descriptions, case-insensitively, first hint that
   matches *any* band wins.
2. **Positional fallback — only for fully unlabelled rasters.** Added in this branch,
   `_POSITIONAL_ROLES` (lines 479–484) is consulted **only if every band description is empty
   or whitespace** (line 513).
3. **Hard failure.** Anything else raises `ValueError` naming the unresolved role.

### 3.2 Hint table

| Role | Hints (priority order) | Physical meaning |
| --- | --- | --- |
| `red` | `B04`, `red`, `r` | ~665 nm |
| `green` | `B03`, `green`, `g` | ~560 nm |
| `nir` | `B08`, `B8A`, `B8`, `nir`, `n` | ~842 nm |
| `swir` | `B11`, `SWIR1`, `B12`, `SWIR2`, `swir` | SWIR1 preferred (Xu 2006 MNDWI) |

### 3.3 Positional table (unlabelled rasters only)

| Band count | Channel → role |
| --- | --- |
| 4 | `1=green, 2=red, 3=nir` (i.e. **B, G, R, NIR**) |
| 5 | `1=green, 2=red, 3=nir, 4=swir` (i.e. **B, G, R, NIR, SWIR1**) |
| 3, 6, 7, … | No fallback — resolve by description or raise |

> [!IMPORTANT]
> This is the **canonical stack convention for the whole project**: 4-band = B, G, R, NIR;
> 5-band = B, G, R, NIR, SWIR1. It matches the positional order already hard-coded in
> `GeospatialEngine.calculate_spectral_indices()` and in the XAI and fusion engines (§4).

### 3.4 Guarantees

For every successful call, `raster_index()` returns `info["band_indices"]` — a dict of
**1-based** channel numbers keyed by role, e.g. `{"nir": 4, "red": 3}` (built at line 536).
This is the value to log in traces and print in reports, because it is the only ground truth
for *which pixels* produced the number.

### 3.5 The guardrail (do not relax it)

The fallback applies **only** to rasters whose descriptions are all empty. A raster that *has*
descriptions which match no hint still fails loudly, with an explanatory suffix. This is
deliberate: silently guessing `green` from position on a labelled, unusually-ordered stack
(e.g. `ch1…ch4`) would produce a numerically plausible but physically wrong index that no
downstream tool could detect. A 4-band unlabelled stack asked for `ndbi` also still raises,
because 4 bands carry no SWIR — the fallback must not invent one.

Both properties are locked by tests in `tests/test_geospatial.py`:
`test_unlabelled_raster_uses_positional_fallback` and
`test_labelled_raster_without_matching_hints_still_raises`.

---

## 4. Codebase map — who assumes what

| Module | Location | Assumption today | Risk from band order | Owner |
| --- | --- | --- | --- | --- |
| `services/geospatial.py` | `calculate_spectral_indices` (770–786) | Hard-codes `green=1, red=2, nir=3`, `swir=4`; returns no provenance | Correct by construction, but **returns no `band_indices`**, so callers can't audit it | Misha |
| `tools/spatial_grounding.py` | `_run_fallback` (299–309) | Calls `calculate_spectral_indices()`, reads `NDVI`/`NDWI` | Inherits positional order invisibly; no band info reaches the trace | Chhavi |
| `tools/fusion_engine.py` | (132–139) | `opt_bands[1],[2],[3]`; `opt_bands[4]` for SWIR | Wrong index on any non-B,G,R stack | Chhavi |
| `origin/feature/achintya-backend-api` | `tools/fusion_engine.py:135` | `nir = opt_bands[3]` hard-coded | Same, and it is **already pushed** | Achintya |
| `mlops/xai_engine.py` | `PhysicsScatteringDecomposer` (398–405, docstring 382–383) | Docstring states `[B, G, R, NIR]`; slices `[1],[2],[3]` | Per-band sensitivity attributed to the wrong role | Peter |
| `benchmarks/faithfulness.py` | `DEFAULT_BANDS_4` / `DEFAULT_BANDS_5` (48–49) | 4 = `B02_Blue…B08_NIR`; **5 = 4 optical + `SAR_C_Band`** | ⚠️ **Conflicts with §3.3**: 5-band here means `+SAR`, not `+SWIR1` | Pradipti |
| `scripts/spectral_pfi_benchmark.py` | line 238 | `band_names=[… B08_NIR, SAR_C_Band]` | Same 5-channel clash | Pradipti |
| `frontend/src/pages/Benchmarks.jsx` | 818, 840 | `['B03_Green', 'B08_NIR']` literal band names | Breaks the moment resolution changes; no `band_indices` field rendered | Vinayak |
| `frontend/src/components/ExplainabilityHUD.jsx` | 213 | Iterates `explanation.spectral_sensitivity` keys | Renders whatever labels XAI emits — must match resolved roles | Vinayak |
| `services/report_generator.py` | 435–441 | PDF table over `xai.spectral_sensitivity` | Proxy vs real index is not labelled in the dossier | Achintya |
| `db/models.py` | `JSON_KEYS` / `bands` blob | Persists raw band *names* only | No resolved roles stored — fine, resolution is derivable, no migration needed | Misha |
| `api/main.py` | `/api/query`, `/api/trace/{id}` (399, 477) | Stores trace via orchestrator | `band_indices` never reaches the jury-facing trace | Achintya |
| `services/orchestrator.py` | `verify_input_compatibility` (123–182) | Audits CRS / resolution / bbox | Band roles are absent from `input_audit` | Peter |
| `tests/test_db.py` | 25, 45, 77 | Asserts 4 named bands round-trip | Stays valid | Pradipti |
| `tests/test_dev_workbench.py` | 98 | Already asserts `band_indices == {"nir": 4, "red": 3}` | Locks the workbench path | Misha |

### ⚠️ The one genuine semantic conflict

`benchmarks/faithfulness.py` and `scripts/spectral_pfi_benchmark.py` treat **channel 5 as SAR**,
while §3.3 defines **channel 5 as SWIR1**. Both are internally consistent, but a 5-channel array
means different things to the PFI harness and to the geospatial resolver. Resolve this before any
PFI weight is quoted in the deck:

- **Preferred:** keep 5 = SWIR1 for optical stacks and give SAR its own explicitly-named
  channel at index ≥ 5 (`["B02_Blue","B03_Green","B04_Red","B08_NIR","B11_SWIR1","SAR_C_Band"]`).
- **Minimum:** rename the PFI defaults to `OPTICAL_BANDS_4` / `OPTICAL_PLUS_SAR_5` so no reader
  assumes one convention covers both.

---

## 5. Required changes per branch

| Branch | Owner | Required change | Blocking? |
| --- | --- | --- | --- |
| `feature/misha-geospatial-v2` | Misha | Merge this branch (see §6 for order) | — |
| `origin/feature/chhavi-ai-models` | Chhavi | Route `_run_fallback` through the resolver (or accept `band_indices`), thread roles into the tool trace; stop assuming `opt_bands[3]` is NIR | **Yes** — before develop |
| `origin/feature/achintya-backend-api` | Achintya | Fix `fusion_engine.py:135` `opt_bands[3]`; persist `band_resolution` into the trace payload so `/api/trace/{id}` shows which channels produced the index | **Yes** — before develop |
| `mlops/xai_engine.py` | Peter | Accept an optional `band_names` / resolution argument instead of asserting `[B,G,R,NIR]`; pass it through to `band_sensitivity_attribution` | **Yes** — per-band claims are otherwise unattributable |
| `benchmarks/`, `scripts/spectral_pfi_benchmark.py` | Pradipti | Settle the 5-channel convention (§4); rename the defaults so the assumption is visible | **Yes** — before benchmark numbers are cited |
| `origin/vinayak-frontend-satqueryAI` | Vinayak | Render `band_indices` + `band_names` from the payload instead of literals at `Benchmarks.jsx:818,840`; audit `Globe3D.jsx` basemaps (§7) | No — follow-up PR is acceptable |
| `feature/misha-geospatial-db` | Misha | 33 commits behind `main`; superseded by `-v2`. Rebase or close | Informational |
| `services/report_generator.py` | Achintya | Label proxy indices (`ndbi_red_proxy`) as proxies in the PDF dossier | No |

---

## 6. Merge order and conflict surface

`feature/misha-geospatial-v2` is **2 ahead / 0 behind** `main`, so it fast-forwards cleanly.

1. **This branch first.** It only *adds* `_POSITIONAL_ROLES`, one guard clause, and two tests —
   no existing symbol changes meaning, so nothing built on it can break by ordering.
2. **Peter (XAI) and Chhavi (tools) next**, since both touch the band-order assumption and will
   want the resolver in place.
3. **Achintya's backend PR** after both — it needs the trace field to have something to carry.
4. **Pradipti's benchmark rename** can land anytime, but must precede any deck that quotes PFI.

Conflict surface is small: `services/geospatial.py` is owned solely by Misha, and the other
branches touch `tools/` and `mlops/`. The only textual overlap is `tools/fusion_engine.py`
between Chhavi's and Achintya's branches — that file needs one agreed edit, not two.

---

## 7. Basemap / tile policy (applies to the workbench *and* the frontend)

**The bug.** The committed workbench used `folium.Map(..., tiles="OpenStreetMap")`. OSM's
volunteer tile servers return **403 "Access blocked"** for pages loaded over `file://`, so every
`data/outputs/dev_workbench/*.html` rendered a wall of 403 tiles.

**The fix.** `scripts/dev_workbench.py:274` now builds the map with `tiles=None` and adds two
explicit keyless layers: **Esri World Imagery** (real satellite context for judging a mask) and
**Esri World Topo (light)**. On a fresh render the footprint, raster overlay, polygons and info
panel are all readable.

**Two rules for everyone:**

1. **No basemap that needs a key, ever.** Carto, Stadia, MapTiler and friends render a tiled
   `API KEY REQUIRED` placeholder *inside* the map when unauthenticated — a result that looks
   like a broken render rather than a credential problem, and which costs an hour to diagnose.
2. **Never rely on a third-party tile host at demo time.** `arcgisonline.com` is keyless, which
   makes it the right default for local development, but Esri's terms restrict these tiles to
   use with Esri products rather than redistribution inside our own deliverable, and the SIH
   venue Wi-Fi may not reach it at all. For the jury build, prefer the pure-raster view (no
   basemap) — the `tiles=None` structure already makes that a one-line change.

> **Action for Vinayak:** `frontend/src/pages/Globe3D.jsx` exposes a satellite/OSM basemap
> switcher (lines 305–424) over Cesium. It has exactly the same failure mode as the workbench
> had, on a much more visible screen. Audit it before the pitch.

**Not part of this PR.** `data/outputs/*` is gitignored (`.gitignore:60`), so the regenerated
`*.html` / `*.json` artefacts are local only — expected, not an oversight.

---

## 8. Definition of done

Misha's side, verified on this branch:

```
.venv/bin/pytest tests/test_geospatial.py      # 48 passed
.venv/bin/pytest tests/test_dev_workbench.py   # 9 passed
```

> Use the **repo-local** `.venv` (`.venv/bin/pytest`). The outer `/Users/misha/Documents/SatQuery/.venv`
> lacks Pillow, so `import services.geospatial` raises `ModuleNotFoundError: No module named 'PIL'`,
> the test modules' guarded import swallows it, and the whole suite reports **silently skipped**
> rather than failing. Do not read "48 skipped" as green.

For your own PR, add:

- **Chhavi** — a tool test proving `_run_fallback` honours a non-standard band order.
- **Peter** — an XAI test where `band_names` is supplied and attributions follow it.
- **Pradipti** — a PFI test pinning the final 5-channel convention.
- **Vinayak** — a render check that band labels come from the payload.

---

## 9. What not to do

- **Do not re-derive band order by hand** in a new tool. Import the resolver, or accept
  `band_indices` as an argument. A second private copy of "index 3 is NIR" is how this class of
  bug comes back.
- **Do not relax the labelled-raster error** (§3.5) to make an awkward file pass. Fix the file's
  band descriptions instead.
- **Do not report a proxy as a real index.** `ndbi_red_proxy` is a fallback for 4-band stacks and
  must be labelled as such wherever it appears.
- **Do not add a keyed tile layer** just to make a demo map look nicer (§7).
