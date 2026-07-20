/* Dashboard behavior: tab switching, tag filters, lightbox, sortable tables.
   No frameworks, no network requests — everything comes from manifest.js and
   data/tables.js, so the page works from file:// as well as any static host. */

(function () {
  "use strict";

  const FIGURES = window.MANIFEST.figures;
  const TABLES = window.MANIFEST.tables;
  const TABLE_DATA = window.TABLE_DATA || {};
  const FIG_META = window.FIG_META || {};

  const FIGURE_TABS = ["deterministic", "probabilistic", "climatology", "events"];
  const ALL_TABS = ["findings", ...FIGURE_TABS, "tables", "about"];
  const FILTER_META = {
    obs:   { label: "Reference" },
    lead:  { label: "Lead day" },
    model: { label: "Model" },
  };

  const el = {
    tabs: document.querySelectorAll(".tab"),
    filterBar: document.getElementById("filter-bar"),
    figurePanel: document.getElementById("figure-panel"),
    findingsPanel: document.getElementById("findings-panel"),
    tablesPanel: document.getElementById("tables-panel"),
    aboutPanel: document.getElementById("about-panel"),
    lightbox: document.getElementById("lightbox"),
    lightboxImg: document.getElementById("lightbox-img"),
    lightboxCaption: document.getElementById("lightbox-caption"),
  };

  const state = { tab: "findings", filters: {} };

  // ── Tabs ──────────────────────────────────────────────────────────────────

  function setTab(tab) {
    state.tab = tab;
    state.filters = {};
    el.tabs.forEach(b => b.classList.toggle("active", b.dataset.tab === tab));

    const isFigureTab = FIGURE_TABS.includes(tab);
    el.figurePanel.hidden = !isFigureTab;
    el.findingsPanel.hidden = tab !== "findings";
    el.tablesPanel.hidden = tab !== "tables";
    el.aboutPanel.hidden = tab !== "about";

    if (isFigureTab) {
      buildFilterBar();
      renderFigures();
    } else {
      el.filterBar.hidden = true;
      el.figurePanel.innerHTML = "";
    }
  }

  // Tabs are routed through the URL hash so every view is linkable and the
  // browser back button walks the tab history.
  el.tabs.forEach(b => b.addEventListener("click", () => {
    location.hash = b.dataset.tab;
  }));

  function applyHash() {
    const tab = location.hash.replace(/^#/, "");
    setTab(ALL_TABS.includes(tab) ? tab : "findings");
  }
  window.addEventListener("hashchange", applyHash);

  // ── Filter bar ────────────────────────────────────────────────────────────

  function tabFigures() {
    return FIGURES.filter(f => f.tab === state.tab);
  }

  function filterDimensions() {
    // Only offer a filter when this tab has >1 distinct value for it.
    const dims = {};
    for (const key of Object.keys(FILTER_META)) {
      const values = [...new Set(tabFigures().flatMap(f => f.tags[key] ? [f.tags[key]] : []))];
      if (values.length > 1) {
        dims[key] = values.sort((a, b) =>
          isFinite(a) && isFinite(b) ? a - b : String(a).localeCompare(String(b)));
      }
    }
    return dims;
  }

  function buildFilterBar() {
    const dims = filterDimensions();
    el.filterBar.innerHTML = "";
    if (!Object.keys(dims).length) { el.filterBar.hidden = true; return; }
    el.filterBar.hidden = false;

    for (const [key, values] of Object.entries(dims)) {
      const group = document.createElement("div");
      group.className = "filter-group";

      const label = document.createElement("span");
      label.className = "filter-label";
      label.textContent = FILTER_META[key].label;
      group.appendChild(label);

      for (const value of ["All", ...values]) {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = value;
        const active = value === "All" ? !state.filters[key] : state.filters[key] === value;
        chip.classList.toggle("active", active);
        chip.addEventListener("click", () => {
          if (value === "All") delete state.filters[key];
          else state.filters[key] = value;
          buildFilterBar();
          renderFigures();
        });
        group.appendChild(chip);
      }
      el.filterBar.appendChild(group);
    }

    const clear = document.createElement("button");
    clear.className = "clear-filters";
    clear.textContent = "✕ Clear filters";
    clear.addEventListener("click", () => {
      state.filters = {};
      buildFilterBar();
      renderFigures();
    });
    el.filterBar.appendChild(clear);

    const count = document.createElement("span");
    count.className = "match-count";
    count.id = "match-count";
    el.filterBar.appendChild(count);
  }

  // ── Figure cards ──────────────────────────────────────────────────────────

  function figureVisible(fig) {
    // A filter only applies to figures parameterized by that dimension;
    // untagged figures (e.g. ACC curves under a lead-day filter) stay visible.
    return Object.entries(state.filters).every(
      ([key, value]) => !fig.tags[key] || fig.tags[key] === value);
  }

  function renderFigures() {
    const figures = tabFigures().filter(figureVisible);
    el.figurePanel.innerHTML = "";

    const count = document.getElementById("match-count");
    if (count) count.textContent = `${figures.length} figure${figures.length === 1 ? "" : "s"}`;

    if (!figures.length) {
      el.figurePanel.innerHTML = "<p class='empty-note'>No figures match the current filters.</p>";
      return;
    }

    for (const fig of figures) {
      const card = document.createElement("article");
      card.className = "card";

      const head = document.createElement("div");
      head.className = "card-head";
      const h3 = document.createElement("h3");
      h3.textContent = fig.title;
      head.appendChild(h3);
      if (!fig.no_pdf) {
        const pdf = document.createElement("a");
        pdf.className = "pdf-btn";
        pdf.href = `figures/${fig.id}.pdf`;
        pdf.setAttribute("download", "");
        pdf.textContent = "⬇ PDF";
        head.appendChild(pdf);
      }
      card.appendChild(head);

      const caption = document.createElement("p");
      caption.className = "caption";
      caption.textContent = fig.caption;
      card.appendChild(caption);

      const img = document.createElement("img");
      img.src = `figures/web/${fig.id}.webp`;
      img.alt = fig.title;
      img.loading = "lazy";
      img.decoding = "async";
      const dims = FIG_META[fig.id];
      if (dims) { img.width = dims[0]; img.height = dims[1]; }
      img.addEventListener("click", () => openLightbox(fig));
      card.appendChild(img);

      el.figurePanel.appendChild(card);
    }
  }

  // ── Lightbox ──────────────────────────────────────────────────────────────

  function openLightbox(fig) {
    el.lightboxImg.src = `figures/web/${fig.id}.webp`;
    el.lightboxImg.alt = fig.title;
    el.lightboxCaption.textContent = `${fig.title} — ${fig.caption}`;
    el.lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    el.lightbox.hidden = true;
    el.lightboxImg.src = "";
    document.body.style.overflow = "";
  }

  el.lightbox.addEventListener("click", closeLightbox);
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !el.lightbox.hidden) closeLightbox();
  });

  // ── Tables ────────────────────────────────────────────────────────────────

  function renderTables() {
    el.tablesPanel.innerHTML = "";

    for (const spec of TABLES.filter(t => t.render)) {
      const data = TABLE_DATA[spec.id];
      const card = document.createElement("article");
      card.className = "card table-card";

      const head = document.createElement("div");
      head.className = "card-head";
      const h3 = document.createElement("h3");
      h3.textContent = spec.title;
      head.appendChild(h3);
      const dl = document.createElement("a");
      dl.className = "pdf-btn";
      dl.href = `tables/${spec.id}.csv`;
      dl.setAttribute("download", "");
      dl.textContent = "⬇ CSV";
      head.appendChild(dl);
      card.appendChild(head);

      if (!data) {
        const note = document.createElement("p");
        note.className = "caption";
        note.textContent = "Table data not embedded yet — run website/sync_outputs.py.";
        card.appendChild(note);
      } else {
        card.appendChild(buildSortableTable(data));
      }
      el.tablesPanel.appendChild(card);
    }

    const downloadOnly = TABLES.filter(t => !t.render);
    if (downloadOnly.length) {
      const card = document.createElement("article");
      card.className = "card table-card";
      const head = document.createElement("div");
      head.className = "card-head";
      const h3 = document.createElement("h3");
      h3.textContent = "Additional tables (download only)";
      head.appendChild(h3);
      card.appendChild(head);

      const list = document.createElement("ul");
      list.className = "download-list";
      for (const spec of downloadOnly) {
        const li = document.createElement("li");
        const desc = document.createElement("span");
        desc.className = "desc";
        desc.textContent = spec.title;
        const a = document.createElement("a");
        a.className = "pdf-btn";
        a.href = `tables/${spec.id}.csv`;
        a.setAttribute("download", "");
        a.textContent = "⬇ CSV";
        li.appendChild(desc);
        li.appendChild(a);
        list.appendChild(li);
      }
      card.appendChild(list);
      el.tablesPanel.appendChild(card);
    }
  }

  function buildSortableTable(data) {
    const scroll = document.createElement("div");
    scroll.className = "table-scroll";
    const table = document.createElement("table");
    table.className = "data";

    const numeric = data.columns.map((_, c) =>
      data.rows.every(r => r[c] === "" || r[c] === null || isFinite(parseFloat(r[c]))));

    let sortCol = null, sortDir = 1;

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    data.columns.forEach((name, c) => {
      const th = document.createElement("th");
      th.addEventListener("click", () => {
        sortDir = sortCol === c ? -sortDir : 1;
        sortCol = c;
        renderBody();
        headRow.querySelectorAll(".arrow").forEach(s => s.remove());
        const arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.textContent = sortDir === 1 ? " ▲" : " ▼";
        th.appendChild(arrow);
      });
      th.textContent = name;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    table.appendChild(tbody);

    function renderBody() {
      const rows = [...data.rows];
      if (sortCol !== null) {
        rows.sort((a, b) => {
          const [x, y] = [a[sortCol], b[sortCol]];
          const cmp = numeric[sortCol]
            ? (parseFloat(x) || 0) - (parseFloat(y) || 0)
            : String(x).localeCompare(String(y));
          return sortDir * cmp;
        });
      }
      tbody.innerHTML = "";
      for (const row of rows) {
        const tr = document.createElement("tr");
        for (const value of row) {
          const td = document.createElement("td");
          td.textContent = value;
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
    }

    renderBody();
    scroll.appendChild(table);
    return scroll;
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  document.querySelectorAll(".prose-fig").forEach(img => {
    img.addEventListener("click", () => {
      el.lightboxImg.src = img.src;
      el.lightboxImg.alt = img.alt;
      el.lightboxCaption.textContent = img.dataset.title || img.alt;
      el.lightbox.hidden = false;
      document.body.style.overflow = "hidden";
    });
  });

  renderTables();
  applyHash();
})();
