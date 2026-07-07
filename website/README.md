# Verification results dashboard

A self-contained static website (no build step, no server, no dependencies)
presenting every figure and table produced by `run_verification.py` for the
MAM 2024 benchmark, plus the narrative content (key findings, scorecard,
methodology, glossary, reproduction commands) formerly on the MkDocs site.
UI modeled on
[RainCheck Africa](https://africlimate-research.github.io/RainCheckAfrica/):
header with headline stats, tab views, filter chips, lightbox zoom.

## View it

Open `index.html` directly in a browser (works from `file://` — all table data
is embedded, nothing is fetched), or serve it:

```bash
python -m http.server 8000 --directory website
# → http://localhost:8000
```

## Refresh after a pipeline rerun

```bash
python run_verification.py          # regenerates mam2024_analysis_outputs/
python website/sync_outputs.py      # PDFs/CSVs mirrored, WebPs generated, tables embedded
```

`sync_outputs.py` needs **Pillow** (its only non-stdlib dependency) to build
the display images: ~1500px WebP (~10× smaller than the 300-dpi PNGs, which
are not shipped — the PDF is the high-resolution artifact). It cross-validates
`manifest.js` against the output folder in both directions and **fails** if
they disagree (renamed/missing/unlisted files), so the site can't silently go
stale. If the pipeline gains or renames a figure, add/update its entry in
`manifest.js`.

## Layout

```
index.html        page skeleton (tabs, panels, lightbox, findings, about)
style.css         design system — palette mirrors benchmark_ea/verification/style.py
app.js            hash-routed tabs, filters, lightbox, sortable tables (vanilla JS)
manifest.js       hand-authored catalog: figure titles/captions/tabs/tags, table list
sync_outputs.py   mirror + convert + validate + embed (run after every pipeline rerun)
figures/*.pdf     vector PDFs, the hi-res download   (generated, committed)
figures/web/      ~1500px WebP display images        (generated, committed)
tables/           raw CSVs                           (generated, committed)
data/tables.js    embedded CSV data                  (generated, committed)
data/figmeta.js   display-image pixel sizes          (generated, committed)
```

Generated outputs are committed so the site is deployable as-is. Only tables
with no figure equivalent are rendered (event scores, Brier, interval
coverage — `render: true` in `manifest.js`); the rest are download-only CSVs.

## Deployment

This folder **is** the site. `.github/workflows/docs.yml` publishes it as-is
to the root of `gh-pages` on every push to `main` touching `website/**`:

- https://ayalaajil.github.io/Benchmark-EastAfrica/

The old MkDocs site is retired; its content lives in the **Key findings** and
**About** tabs here. Don't push to `gh-pages` directly — the workflow
force-overwrites it.
