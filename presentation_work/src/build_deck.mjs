import fs from "node:fs";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  panel,
  text,
  image,
  shape,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} from "@oai/artifact-tool";

const ROOT = "C:/SeniorProj";
const OUT_DIR = "C:/SeniorProj/presentation_work/output";
const SCRATCH_DIR = "C:/SeniorProj/presentation_work/scratch";
const PREVIEW_DIR = path.join(SCRATCH_DIR, "previews");
const LAYOUT_DIR = path.join(SCRATCH_DIR, "layouts");
const PPTX_PATH = path.join(OUT_DIR, "output.pptx");

for (const dir of [OUT_DIR, SCRATCH_DIR, PREVIEW_DIR, LAYOUT_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

function pngDataUrl(filePath) {
  const data = fs.readFileSync(filePath);
  return `data:image/png;base64,${data.toString("base64")}`;
}

const W = 1920;
const H = 1080;
const C = {
  bg: "#F8F6F0",
  ink: "#18212B",
  muted: "#18212B",
  faint: "#D8DED9",
  accent: "#2F7D6D",
  accent2: "#B07A2B",
  blue: "#446B8C",
  surface: "#FFFFFF",
  surface2: "#EEF3EF",
  line: "#C7D1CC",
  good: "#2F7D6D",
  warn: "#B07A2B",
  orange: "#C15D3A",
};

const styles = {
  kicker: { fontSize: 19, bold: true, color: C.accent },
  title: { fontSize: 54, bold: true, color: C.ink },
  subtitle: { fontSize: 30, color: C.ink },
  body: { fontSize: 31, color: C.ink },
  bodySmall: { fontSize: 27, color: C.ink },
  label: { fontSize: 21, bold: true, color: C.ink },
  metric: { fontSize: 56, bold: true, color: C.accent },
  mono: { fontSize: 28, color: C.ink },
  foot: { fontSize: 14, color: C.ink },
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addNotes(slide, note) {
  slide.speakerNotes.setText(note.trim());
}

function t(value, opts = {}) {
  return text(value, {
    width: opts.width ?? fill,
    height: opts.height ?? hug,
    style: opts.style ?? styles.body,
    name: opts.name,
    columnSpan: opts.columnSpan,
    rowSpan: opts.rowSpan,
  });
}

function bulletList(items, opts = {}) {
  return column(
    { name: opts.name, width: opts.width ?? fill, height: hug, gap: opts.gap ?? 18 },
    items.map((item, idx) =>
      t(`- ${item}`, {
        name: `${opts.name ?? "bullets"}-${idx + 1}`,
        style: opts.style ?? styles.bodySmall,
      }),
    ),
  );
}

function chip(label, color = C.accent, width = 260) {
  return panel(
    {
      name: `chip-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
      width: fixed(width),
      height: fixed(56),
      padding: { x: 20, y: 12 },
      fill: "#FFFFFF",
      line: { color, width: 1.5 },
      borderRadius: "rounded-full",
      align: "center",
      justify: "center",
    },
    t(label, {
      width: fill,
      style: { fontSize: 18, bold: true, color, alignment: "center" },
    }),
  );
}

function stat(value, label, color = C.accent) {
  return column(
    { name: `stat-${label}`, width: fill, height: hug, gap: 6 },
    [
      t(value, { style: { ...styles.metric, color }, name: `stat-value-${label}` }),
      t(label, { style: styles.label, name: `stat-label-${label}` }),
    ],
  );
}

function sectionTitle(slide, number, title, subtitle) {
  return column(
    { name: "title-stack", width: fill, height: hug, gap: 10 },
    [
      row(
        { name: "kicker-row", width: fill, height: hug, gap: 18, align: "center" },
        [
          t(String(number).padStart(2, "0"), {
            width: fixed(70),
            style: { fontSize: 18, bold: true, color: C.accent, alignment: "center" },
            name: "slide-number",
          }),
          rule({ name: "kicker-rule", width: fixed(120), stroke: C.accent, weight: 3 }),
          t("Mini Spotify-like Music Recommendation System", {
            width: fill,
            style: styles.kicker,
            name: "deck-kicker",
          }),
        ],
      ),
      t(title, { width: fill, height: fixed(96), style: styles.title, name: "slide-title" }),
      subtitle
        ? t(subtitle, { width: fixed(1460), style: styles.subtitle, name: "slide-subtitle" })
        : null,
    ].filter(Boolean),
  );
}

function addSlide({ number, title, subtitle, body, visual, note, source }) {
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  slide.compose(
    column(
      {
        name: "slide-root",
        width: fill,
        height: fill,
        padding: { x: 88, y: 64 },
        gap: 26,
      },
      [
        sectionTitle(slide, number, title, subtitle),
        body && visual
          ? grid(
              {
                name: "body-grid",
                width: fill,
                height: grow(1),
                columns: [fr(0.96), fr(1.04)],
                columnGap: 56,
                alignItems: "start",
              },
              [body, visual],
            )
          : body || visual,
        row(
          { name: "footer", width: fill, height: hug, align: "center", gap: 20 },
          [
            rule({ name: "footer-rule", width: fixed(120), stroke: C.line, weight: 1 }),
            t(source ?? "Senior Project Presentation", {
              width: fill,
              style: styles.foot,
              name: "source-rail",
            }),
          ],
        ),
      ].filter(Boolean),
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  addNotes(slide, note);
  return slide;
}

function simpleFlow(labels, colors = [C.accent, C.blue, C.orange], opts = {}) {
  const stepWidth = opts.stepWidth ?? (labels.length >= 4 ? 150 : 190);
  const arrowWidth = opts.arrowWidth ?? 30;
  const gap = opts.gap ?? 8;
  return row(
    { name: "flow", width: fill, height: hug, gap, align: "center" },
    labels.flatMap((label, idx) => {
      const node = panel(
        {
          name: `flow-step-${idx + 1}`,
          width: fixed(stepWidth),
          height: fixed(108),
          padding: { x: 12, y: 18 },
          fill: "#FFFFFF",
          line: { color: colors[idx % colors.length], width: 1.5 },
          borderRadius: 10,
          align: "center",
          justify: "center",
        },
        t(label, {
          width: fill,
          style: { fontSize: opts.fontSize ?? 22, bold: true, color: C.ink, alignment: "center" },
        }),
      );
      if (idx === labels.length - 1) return [node];
      return [
        node,
        t("->", {
          width: fixed(arrowWidth),
          style: { fontSize: 26, bold: true, color: C.ink, alignment: "center" },
        }),
      ];
    }),
  );
}

function diagramBox(label, opts = {}) {
  return panel(
    {
      name: opts.name,
      width: opts.width ?? fixed(190),
      height: opts.height ?? fixed(96),
      padding: opts.padding ?? { x: 14, y: 14 },
      fill: opts.fill ?? "#FFFFFF",
      line: { color: opts.color ?? C.accent, width: opts.lineWidth ?? 1.5 },
      borderRadius: opts.radius ?? 10,
      align: "center",
      justify: "center",
    },
    t(label, {
      width: fill,
      style: {
        fontSize: opts.fontSize ?? 22,
        bold: opts.bold ?? true,
        color: opts.textColor ?? C.ink,
        alignment: "center",
      },
    }),
  );
}

function arrow(label = "->", width = 40) {
  return t(label, {
    width: fixed(width),
    style: { fontSize: 26, bold: true, color: C.ink, alignment: "center" },
  });
}

function metricCell(value, opts = {}) {
  return panel(
    {
      name: opts.name,
      width: fill,
      height: fixed(52),
      padding: { x: 10, y: 10 },
      fill: opts.fill ?? "transparent",
      line: { color: opts.line ?? C.faint, width: opts.lineWidth ?? 0.8 },
      borderRadius: 7,
      align: "center",
      justify: "center",
    },
    t(value, {
      width: fill,
      style: {
        fontSize: opts.fontSize ?? 20,
        bold: opts.bold ?? false,
        color: opts.color ?? C.ink,
        alignment: opts.align ?? "center",
      },
    }),
  );
}

function resultsTable() {
  const rows = [
    ["Precision@10", "0.0338", "0.1120", "0.0269", "CF"],
    ["Recall@10", "0.0227", "0.0793", "0.0198", "CF"],
    ["Hit Rate@10", "0.2679", "0.6544", "0.2283", "CF"],
    ["Coverage@10", "0.0035", "0.3949", "0.1235", "CF"],
    ["Novelty@10", "2.4491", "4.4614", "6.2362", "CB"],
    ["Diversity@10", "0.8382", "0.8794", "0.6706", "CF"],
  ];
  const header = ["Metric", "Popularity", "CF (ALS)", "Content-Based", "Winner"];
  const colWidths = [fixed(190), fixed(150), fixed(150), fixed(170), fixed(105)];
  return column(
    { name: "results-table", width: fill, height: hug, gap: 8 },
    [
      grid(
        { name: "results-header", width: fill, height: hug, columns: colWidths, columnGap: 8 },
        header.map((h, idx) =>
          metricCell(h, {
            name: `results-header-${idx}`,
            fill: idx === 2 ? C.accent : idx === 3 ? C.orange : C.surface2,
            line: idx === 2 ? C.accent : idx === 3 ? C.orange : C.line,
            color: idx === 2 || idx === 3 ? "#FFFFFF" : C.ink,
            bold: true,
            fontSize: 18,
          }),
        ),
      ),
      ...rows.map((r, rowIdx) =>
        grid(
          { name: `results-row-${rowIdx + 1}`, width: fill, height: hug, columns: colWidths, columnGap: 8 },
          r.map((cell, colIdx) => {
            const winner = r[4] === "CF" ? 2 : 3;
            const isWinnerValue = colIdx === winner;
            const isWinnerLabel = colIdx === 4;
            return metricCell(cell, {
              name: `results-row-${rowIdx + 1}-cell-${colIdx + 1}`,
              fill: isWinnerValue ? "#E4F1EC" : isWinnerLabel ? "#F4EFE7" : "#FFFFFF",
              line: isWinnerValue ? C.accent : isWinnerLabel ? C.orange : C.faint,
              bold: colIdx === 0 || isWinnerValue || isWinnerLabel,
              color: isWinnerValue ? C.accent : isWinnerLabel ? C.orange : C.ink,
              align: colIdx === 0 ? "left" : "center",
              fontSize: colIdx === 0 ? 18 : 19,
            });
          }),
        ),
      ),
    ],
  );
}

function miniMatrix(name, rows, cols, active, color = C.accent, cell = 34) {
  return grid(
    {
      name,
      width: fixed(cols * cell + (cols - 1) * 6),
      height: fixed(rows * cell + (rows - 1) * 6),
      columns: Array.from({ length: cols }, () => fr(1)),
      rows: Array.from({ length: rows }, () => fr(1)),
      columnGap: 6,
      rowGap: 6,
    },
    Array.from({ length: rows * cols }, (_, i) =>
      shape({
        name: `${name}-cell-${i}`,
        width: fill,
        height: fill,
        fill: active.includes(i) ? color : C.surface2,
        line: { color: C.line, width: 1 },
        borderRadius: 5,
      }),
    ),
  );
}

function cover() {
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  slide.compose(
    grid(
      {
        name: "cover-root",
        width: fill,
        height: fill,
        padding: { x: 110, y: 76 },
        columns: [fr(1.08), fr(0.92)],
        rows: [fr(1), auto],
        columnGap: 84,
        rowGap: 30,
        alignItems: "center",
      },
      [
        column(
          { name: "cover-lockup", width: fill, height: hug, gap: 24 },
          [
            t("Senior Project", { style: { ...styles.kicker, fontSize: 19 }, name: "cover-kicker" }),
            t("Mini Spotify-like\nMusic Recommendation\nSystem", {
              width: wrap(820),
              style: { fontSize: 76, bold: true, color: C.ink },
              name: "cover-title",
            }),
            rule({ name: "cover-rule", width: fixed(280), stroke: C.accent, weight: 6 }),
            t("Comparing popularity, collaborative filtering, and tag-based recommendation as prediction and discovery systems.", {
              width: wrap(780),
              style: { fontSize: 29, color: C.muted },
              name: "cover-promise",
            }),
          ],
        ),
        column(
          { name: "cover-question", width: fill, height: hug, gap: 28 },
          [
            t("How does a music platform decide what song to recommend next?", {
              width: wrap(680),
              style: { fontSize: 42, bold: true, color: C.accent },
              name: "cover-question-text",
            }),
            simpleFlow(["Popular", "Similar users", "Similar songs"], [C.blue, C.accent, C.orange], { stepWidth: 178 }),
          ],
        ),
        row(
          { name: "cover-footer", columnSpan: 2, width: fill, height: hug, align: "center" },
          [
            t("Presenter: [Your Name]  |  Course: [Course Name]  |  2026", {
              width: fill,
              style: styles.foot,
              name: "cover-footer-text",
            }),
          ],
        ),
      ],
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  addNotes(
    slide,
    `Open with the core question: what should a music platform recommend next? Explain that the project compares globally popular songs, songs liked by similar users, and songs similar to a user's history. Emphasize that recommendation systems shape discovery, not only prediction.`,
  );
}

cover();

addSlide({
  number: 2,
  title: "Dataset and problem type",
  subtitle: "KGRec-music is an implicit-feedback ranking problem, not a star-rating prediction task.",
  body: column({ name: "dataset-body", width: fill, height: hug, gap: 22 }, [
    row({ name: "dataset-stats", width: fill, height: hug, gap: 44 }, [
      stat("5,199", "users", C.accent),
      stat("8,640", "music items", C.blue),
      stat("751,531", "interactions", C.orange),
    ]),
    bulletList(
      [
        "Each interaction is marked as 1: observed behavior, not satisfaction.",
        "The model ranks candidate songs instead of predicting ratings.",
        "Success means held-out interacted songs appear near the top.",
      ],
      { name: "dataset-bullets" },
    ),
  ]),
  visual: panel(
    {
      name: "ranking-visual",
      width: fill,
      height: fixed(420),
      padding: { x: 34, y: 34 },
      fill: "#FFFFFF",
      line: { color: C.line, width: 1 },
      borderRadius: 12,
    },
    column({ width: fill, height: fill, gap: 24, justify: "center" }, [
      t("Top-K ranking protocol", { style: { ...styles.label, color: C.accent } }),
      row({ name: "ranking-protocol", width: fill, height: hug, gap: 14, align: "center" }, [
        diagramBox("Partial user\nhistory", { color: C.blue, width: fixed(164), height: fixed(104), fontSize: 21 }),
        arrow(),
        diagramBox("Rank unseen\ncandidate songs", { color: C.accent, width: fixed(190), height: fixed(104), fontSize: 20 }),
        arrow(),
        diagramBox("Top-K\nrecommendations", { color: C.orange, width: fixed(178), height: fixed(104), fontSize: 20 }),
      ]),
      row({ name: "ranking-check", width: fill, height: hug, gap: 14, align: "center" }, [
        diagramBox("Held-out\ninteractions", { color: C.orange, fill: "#FFF8F3", width: fixed(180), height: fixed(96), fontSize: 20 }),
        arrow("?", 36),
        t("Do they appear near the top of the ranked list?", {
          width: wrap(520),
          style: { fontSize: 28, color: C.ink, bold: true },
        }),
      ]),
    ]),
  ),
  note:
    "Explain that KGRec-music contains user-item interactions, tags, and descriptions. Because the interactions are implicit, the project asks a ranking question: given part of a user's history, can the recommender place later interacted items near the top?",
  source: "Source: KGRec-music preprocessing summary and final_report.txt",
});

addSlide({
  number: 3,
  title: "Preprocessing defines the experiment",
  subtitle: "Cleaning is not a side task; it creates the matrix, feature space, and reproducible artifact boundary.",
  body: bulletList(
    [
      "Map raw user and item IDs into continuous internal indices.",
      "Preserve reverse mapping so predictions still point back to KGRec item IDs.",
      "Build item feature vectors for both content scoring and diversity evaluation.",
      "Write frozen train, validation, test, and feature artifacts to disk.",
    ],
    { name: "preprocess-bullets", style: styles.body, gap: 14 },
  ),
  visual: panel(
    {
      name: "preprocess-map-panel",
      width: fill,
      height: fixed(500),
      padding: { x: 32, y: 30 },
      fill: "#FFFFFF",
      line: { color: C.line, width: 1 },
      borderRadius: 12,
    },
    column({ name: "id-flow", width: fill, height: fill, gap: 24, justify: "center" }, [
      row({ width: fill, height: hug, gap: 14, align: "center" }, [
        t("Users", { width: fixed(80), style: { fontSize: 22, bold: true, color: C.blue } }),
        diagramBox("raw user IDs", { color: C.blue, width: fixed(158), height: fixed(88), fontSize: 20 }),
        arrow(),
        diagramBox("0..5198", { color: C.accent, width: fixed(136), height: fixed(88), fontSize: 22 }),
        arrow(),
        diagramBox("matrix rows", { color: C.orange, width: fixed(158), height: fixed(88), fontSize: 20 }),
      ]),
      row({ width: fill, height: hug, gap: 14, align: "center" }, [
        t("Items", { width: fixed(80), style: { fontSize: 22, bold: true, color: C.blue } }),
        diagramBox("raw item IDs", { color: C.blue, width: fixed(158), height: fixed(88), fontSize: 20 }),
        arrow(),
        diagramBox("0..8639", { color: C.accent, width: fixed(136), height: fixed(88), fontSize: 22 }),
        arrow(),
        diagramBox("matrix columns", { color: C.orange, width: fixed(158), height: fixed(88), fontSize: 20 }),
      ]),
      rule({ width: fill, stroke: C.line, weight: 1 }),
      row({ width: fill, height: hug, gap: 18, align: "center" }, [
        diagramBox("ID maps", { color: C.accent, fill: C.surface2, width: fixed(168), height: fixed(86), fontSize: 22 }),
        t("Forward maps make training efficient; reverse maps make outputs explainable and usable.", {
          width: wrap(560),
          style: { fontSize: 27, color: C.ink },
          name: "preprocess-caption",
        }),
      ]),
    ]),
  ),
  note:
    "Frame preprocessing as the step that defines the entire experiment. Raw IDs are mapped into continuous indices for matrix models like ALS, while mappings back to original KGRec IDs are preserved so recommendations remain meaningful outside the model.",
});

addSlide({
  number: 4,
  title: "A fair per-user split",
  subtitle: "No timestamps means the evaluation uses a deterministic random split for each user's interactions.",
  body: column({ name: "split-body", width: fill, height: hug, gap: 28 }, [
    bulletList(
      [
        "Train: 80% of each user's interactions.",
        "Validation: 10% for model development and reporting.",
        "Test: 10% reserved for final evaluation.",
        "All three models share the same split and candidate rules.",
      ],
      { name: "split-bullets", style: styles.bodySmall },
    ),
  ]),
  visual: column({ name: "split-visual", width: fill, height: hug, gap: 20 }, [
    row({ name: "split-bar", width: fill, height: fixed(88), gap: 0 }, [
      panel({ width: grow(8), height: fill, fill: C.accent, justify: "center", align: "center" }, t("80% train", { style: { fontSize: 28, bold: true, color: "#FFFFFF", alignment: "center" } })),
      panel({ width: grow(1), height: fill, fill: C.blue, justify: "center", align: "center" }, t("10% val", { style: { fontSize: 23, bold: true, color: "#FFFFFF", alignment: "center" } })),
      panel({ width: grow(1), height: fill, fill: C.orange, justify: "center", align: "center" }, t("10% test", { style: { fontSize: 23, bold: true, color: "#FFFFFF", alignment: "center" } })),
    ]),
    t("Evaluation asks how well the model understands known users from partial histories.", {
      width: wrap(760),
      style: { fontSize: 30, bold: true, color: C.ink },
    }),
  ]),
  note:
    "Explain why chronological splitting would be ideal but impossible without timestamps. The per-user random 80/10/10 split lets the model learn from part of each user's history and recover held-out interactions. The comparison is fair because all models use the same split.",
});

addSlide({
  number: 5,
  title: "Leakage control keeps the results honest",
  subtitle: "Validation and test information must not influence training-time features or model fitting.",
  body: bulletList(
    [
      "All popularity counts, ALS factors, and content profiles are learned from training data.",
      "TF-IDF vocabulary and tag weights are fitted without using held-out interactions.",
      "Validation/test data are used only as ground truth during evaluation.",
    ],
    { name: "leakage-bullets", style: styles.bodySmall },
  ),
  visual: column({ name: "leakage-visual", width: fill, height: hug, gap: 22 }, [
    row({ name: "leakage-row", width: fill, height: hug, gap: 18, align: "center" }, [
      panel({ width: fixed(300), height: fixed(128), padding: 18, fill: "#FFFFFF", line: { color: C.accent, width: 2 }, borderRadius: 10, justify: "center" }, t("Training data\nfit models + TF-IDF", { style: { fontSize: 26, bold: true, color: C.ink, alignment: "center" } })),
      t("||", { width: fixed(54), style: { fontSize: 44, bold: true, color: C.orange, alignment: "center" } }),
      panel({ width: fixed(300), height: fixed(128), padding: 18, fill: "#FFFFFF", line: { color: C.orange, width: 2 }, borderRadius: 10, justify: "center" }, t("Held-out data\nevaluate only", { style: { fontSize: 26, bold: true, color: C.ink, alignment: "center" } })),
    ]),
    t("The boundary prevents the model from indirectly seeing the answers.", {
      width: wrap(760),
      style: { fontSize: 29, color: C.ink },
    }),
  ]),
  note:
    "Introduce data leakage as a major risk in recommender evaluation. In this project, training data defines the models and feature spaces; validation and test data remain hidden until evaluation. Mention TF-IDF specifically because it learns a vocabulary and weights.",
});

addSlide({
  number: 6,
  title: "Candidate pool and top-K evaluation",
  subtitle: "The system ranks unseen candidate items, then checks whether held-out interactions appear in the top K.",
  body: bulletList(
    [
      "Start from the training catalog.",
      "Remove items the user already interacted with in training.",
      "Rank the remaining candidates for K = 5, 10, and 20.",
      "Compare the recommendation list with validation or test ground truth.",
    ],
    { name: "candidate-bullets", style: styles.bodySmall },
  ),
  visual: panel(
    {
      name: "candidate-eval-panel",
      width: fill,
      height: fixed(500),
      padding: { x: 30, y: 28 },
      fill: "#FFFFFF",
      line: { color: C.line, width: 1 },
      borderRadius: 12,
    },
    column({ name: "candidate-visual", width: fill, height: fill, gap: 24, justify: "center" }, [
      row({ width: fill, height: hug, gap: 14, align: "center" }, [
        diagramBox("Training\ncatalog", { color: C.blue, width: fixed(154), height: fixed(92), fontSize: 20 }),
        arrow("- seen", 74),
        diagramBox("Candidate\npool", { color: C.accent, width: fixed(154), height: fixed(92), fontSize: 20 }),
        arrow("rank", 60),
        diagramBox("Top-K\nlist", { color: C.orange, width: fixed(138), height: fixed(92), fontSize: 20 }),
      ]),
      grid(
        {
          name: "candidate-rank-grid",
          width: fill,
          height: hug,
          columns: [fr(1), fr(1), fr(1), fr(1), fr(1)],
          columnGap: 10,
        },
        ["#1", "#2", "#3", "...", "#K"].map((label, idx) =>
          metricCell(label, {
            fill: idx < 3 ? "#E4F1EC" : C.surface2,
            line: idx < 3 ? C.accent : C.line,
            bold: true,
            color: idx < 3 ? C.accent : C.muted,
          }),
        ),
      ),
      row({ width: fill, height: hug, gap: 16, align: "center" }, [
        diagramBox("Validation / test\nheld-out items", { color: C.orange, fill: "#FFF8F3", width: fixed(236), height: fixed(90), fontSize: 20 }),
        arrow("compare", 90),
        t("Credit only comes from retrieving songs hidden from the model.", {
          width: wrap(440),
          style: { fontSize: 26, bold: true, color: C.ink },
        }),
      ]),
    ]),
  ),
  note:
    "Explain the candidate pool logic: for each user, remove training interactions so the model cannot recommend already-seen items. The held-out split is then the ground truth. This makes the task a top-K ranking problem.",
});

addSlide({
  number: 7,
  title: "Model 1: Popularity baseline",
  subtitle: "A simple deterministic floor: recommend what received the most training interactions.",
  body: bulletList(
    [
      "Score item i by its global training interaction count.",
      "Filter each user's already-seen training items.",
      "Almost the same ranking for every user.",
      "Useful baseline, but weak personalization and narrow exposure.",
    ],
    { name: "pop-bullets", style: styles.bodySmall },
  ),
  visual: column({ name: "pop-visual", width: fill, height: hug, gap: 18 }, [
    t("Mental model", { style: styles.label }),
    t("\"I do not know your specific taste, so I will recommend what many people already interacted with.\"", {
      width: wrap(760),
      style: { fontSize: 35, bold: true, color: C.accent },
      name: "pop-quote",
    }),
    row({ width: fill, height: hug, gap: 12 }, [
      chip("simple", C.blue, 180),
      chip("deterministic", C.accent, 230),
      chip("low novelty", C.orange, 220),
    ]),
  ]),
  note:
    "Describe popularity as the performance floor. It counts training interactions and recommends the most popular remaining items. Its strength is simplicity; its weakness is that it repeatedly recommends the same mainstream items and barely personalizes.",
});

addSlide({
  number: 8,
  title: "Model 2: ALS collaborative filtering",
  subtitle: "ALS learns hidden user and item vectors from the interaction matrix.",
  body: column({ name: "als-body", width: fill, height: hug, gap: 18 }, [
    bulletList(
      [
        "Uses behavior patterns rather than song tags or titles.",
        "Learns latent factors for users and items.",
        "Scores a pair by vector alignment.",
        "Strong when co-listening structure is rich.",
      ],
      { name: "als-bullets", style: styles.bodySmall },
    ),
    t("score(u, i) = dot(x_u, y_i)", {
      width: fixed(720),
      style: { fontSize: 40, bold: true, color: C.accent },
      name: "als-formula",
    }),
  ]),
  visual: panel(
    {
      name: "als-matrix-panel",
      width: fill,
      height: fixed(480),
      padding: { x: 36, y: 30 },
      fill: "#FFFFFF",
      line: { color: C.line, width: 1 },
      borderRadius: 12,
    },
    column({ width: fill, height: fill, gap: 24, justify: "center" }, [
      row({ name: "als-factorization", width: fill, height: hug, gap: 22, align: "center" }, [
        column({ width: hug, height: hug, gap: 5, align: "center" }, [
          t("R", { width: fixed(130), style: { fontSize: 26, bold: true, color: C.blue, alignment: "center" } }),
          miniMatrix("als-r-matrix", 4, 5, [0, 4, 7, 8, 13, 16], C.accent, 34),
          t("user-item interactions", { width: fixed(200), style: { fontSize: 17, color: C.ink, alignment: "center" } }),
        ]),
        t("~", { width: fixed(34), style: { fontSize: 44, bold: true, color: C.ink, alignment: "center" } }),
        column({ width: hug, height: hug, gap: 5, align: "center" }, [
          t("X", { width: fixed(120), style: { fontSize: 26, bold: true, color: C.blue, alignment: "center" } }),
          miniMatrix("als-x-matrix", 4, 2, [0, 2, 5, 7], C.blue, 36),
          t("user factors", { width: fixed(150), style: { fontSize: 17, color: C.ink, alignment: "center" } }),
        ]),
        t("x", { width: fixed(30), style: { fontSize: 34, bold: true, color: C.ink, alignment: "center" } }),
        column({ width: hug, height: hug, gap: 5, align: "center" }, [
          t("Y^T", { width: fixed(160), style: { fontSize: 26, bold: true, color: C.orange, alignment: "center" } }),
          miniMatrix("als-y-matrix", 2, 5, [0, 3, 5, 6, 9], C.orange, 36),
          t("item factors", { width: fixed(160), style: { fontSize: 17, color: C.ink, alignment: "center" } }),
        ]),
      ]),
      rule({ width: fill, stroke: C.line, weight: 1 }),
      row({ width: fill, height: hug, gap: 14, align: "center" }, [
        diagramBox("new user\nvector x_u", { color: C.blue, width: fixed(182), height: fixed(82), fontSize: 19 }),
        arrow("dot", 50),
        diagramBox("item\nvector y_i", { color: C.orange, width: fixed(170), height: fixed(82), fontSize: 19 }),
        arrow("=", 34),
        t("score(u, i)", { width: fixed(176), style: { fontSize: 29, bold: true, color: C.accent, alignment: "center" } }),
      ]),
    ]),
  ),
  note:
    "Explain ALS as the strongest project model. It uses the user-item matrix to learn latent factors. These hidden dimensions are not directly labeled, but they capture behavior patterns such as users who liked item A often liking item B. Missing interactions are treated as low-confidence unknowns, not strong dislikes.",
});

addSlide({
  number: 9,
  title: "Model 3: Content-based recommendation using tags",
  subtitle: "The model recommends songs whose tag vectors are close to the user's tag profile.",
  body: bulletList(
    [
      "Convert item tags into TF-IDF vectors.",
      "Build a user profile from interacted songs.",
      "Rank candidates by cosine similarity in tag space.",
      "More explainable and novel, but tag similarity is not always user preference.",
    ],
    { name: "cb-bullets", style: styles.bodySmall },
  ),
  visual: panel(
    {
      name: "cb-tag-panel",
      width: fill,
      height: fixed(500),
      padding: { x: 30, y: 28 },
      fill: "#FFFFFF",
      line: { color: C.line, width: 1 },
      borderRadius: 12,
    },
    column({ name: "cb-visual", width: fill, height: fill, gap: 22, justify: "center" }, [
      row({ width: fill, height: hug, gap: 9, align: "center" }, [
        diagramBox("seed songs\nwith tags", { color: C.blue, width: fixed(158), height: fixed(94), fontSize: 20 }),
        arrow(),
        diagramBox("TF-IDF\nitem vectors", { color: C.accent, width: fixed(158), height: fixed(94), fontSize: 20 }),
        arrow(),
        diagramBox("mean user\ntag profile", { color: C.orange, width: fixed(168), height: fixed(94), fontSize: 20 }),
        arrow(),
        diagramBox("cosine\nsimilarity", { color: C.accent, width: fixed(148), height: fixed(94), fontSize: 20 }),
      ]),
      grid(
        { name: "tag-example-grid", width: fill, height: hug, columns: [fr(1), fr(1), fr(1)], columnGap: 12 },
        [
          metricCell("indie", { fill: "#EEF3EF", line: C.accent, color: C.accent, bold: true }),
          metricCell("mellow", { fill: "#EEF3EF", line: C.accent, color: C.accent, bold: true }),
          metricCell("80s", { fill: "#EEF3EF", line: C.accent, color: C.accent, bold: true }),
        ],
      ),
      panel(
        { width: fill, height: hug, padding: { x: 22, y: 18 }, fill: C.surface2, line: { color: C.line, width: 1 }, borderRadius: 10 },
        t("Clear explanation: recommend songs sharing high-weight tags. Trade-off: similar tags can create a narrow list.", {
          style: { fontSize: 26, bold: true, color: C.ink },
        }),
      ),
    ]),
  ),
  note:
    "Describe content-based recommendation as item-feature driven. It uses tags, builds a profile from songs the user interacted with, and scores candidates by similarity. It is explainable and can reach long-tail items, but tags may be missing, noisy, or too broad.",
});

addSlide({
  number: 10,
  title: "Metrics: accuracy plus discovery",
  subtitle: "A music recommender should recover relevant songs and shape useful discovery.",
  body: column({ name: "metrics-left", width: fill, height: hug, gap: 20 }, [
    t("Accuracy metrics", { style: { ...styles.label, color: C.accent } }),
    bulletList(
      [
        "Precision@K: how many recommendations were relevant.",
        "Recall@K: how many held-out items were recovered.",
        "NDCG@K: whether relevant items appear near the top.",
        "Hit Rate@K: whether at least one relevant item appears.",
      ],
      { name: "accuracy-metrics", style: styles.bodySmall, gap: 12 },
    ),
  ]),
  visual: column({ name: "metrics-right", width: fill, height: hug, gap: 20 }, [
    t("Beyond-accuracy metrics", { style: { ...styles.label, color: C.orange } }),
    bulletList(
      [
        "Coverage: how much of the catalog is ever recommended.",
        "Novelty: whether recommendations are less obvious.",
        "Diversity: how different items are within one list.",
      ],
      { name: "beyond-metrics", style: styles.bodySmall, gap: 12 },
    ),
    t("Accuracy alone does not tell us what listening experience the model creates.", {
      width: wrap(720),
      style: { fontSize: 29, bold: true, color: C.ink },
    }),
  ]),
  note:
    "Walk through the two metric families. Accuracy measures retrieval of held-out interactions; beyond-accuracy measures exposure, discovery, and variety. The motivation is that music recommendation should not only predict what users already know.",
});

addSlide({
  number: 11,
  title: "Results at K = 10",
  subtitle: "ALS wins the main accuracy metrics and also broadens catalog exposure; content-based wins novelty.",
  body: column({ name: "results-left", width: fill, height: hug, gap: 18 }, [
    t("One-read interpretation", { style: { ...styles.label, color: C.accent } }),
    bulletList(
      [
        "ALS is the best overall ranking model.",
        "Popularity is useful but concentrates exposure on a tiny catalog slice.",
        "Content-based recommendation reaches the long tail, but loses accuracy.",
        "Novelty and diversity are different: obscure does not always mean varied.",
      ],
      { name: "results-bullets", style: styles.bodySmall, gap: 14 },
    ),
    row({ width: fill, height: hug, gap: 16 }, [
      chip("CF wins accuracy", C.accent, 250),
      chip("CB wins novelty", C.orange, 230),
    ]),
  ]),
  visual: resultsTable(),
  note:
    "Interpret the results, not every number. At K=10, ALS has the strongest precision, recall, NDCG, hit rate, coverage, and diversity. Popularity is useful but narrow. Content-based has the highest novelty, showing it reaches more long-tail items, but novelty does not automatically mean diversity.",
  source: "Validation results at K = 10 from artifacts/results/final_report.txt and comparison_table.md",
});

addSlide({
  number: 12,
  title: "Product extension: live demo layer",
  subtitle: "The demo turns the research pipeline into an interactive music recommendation prototype.",
  body: bulletList(
    [
      "New website users are not original KGRec training users.",
      "Fold-in estimates a temporary user vector from seed songs and tags.",
      "Controls expose content weight, novelty, diversity, and list length.",
      "NetEase enrichment is display-only: title, artist, cover, and link.",
    ],
    { name: "demo-bullets", style: styles.bodySmall },
  ),
  visual: panel(
    {
      name: "demo-architecture-panel",
      width: fill,
      height: fixed(520),
      padding: { x: 30, y: 28 },
      fill: "#FFFFFF",
      line: { color: C.line, width: 1 },
      borderRadius: 12,
    },
    column({ width: fill, height: fill, gap: 24, justify: "center" }, [
      row({ name: "demo-main-flow", width: fill, height: hug, gap: 12, align: "center" }, [
        diagramBox("User input\nsongs + tags", { color: C.blue, width: fixed(150), height: fixed(98), fontSize: 19 }),
        arrow(),
        diagramBox("Fold-in\ntaste vector", { color: C.accent, width: fixed(150), height: fixed(98), fontSize: 19 }),
        arrow(),
        diagramBox("Rank + rerank\ncandidates", { color: C.orange, width: fixed(162), height: fixed(98), fontSize: 19 }),
        arrow(),
        diagramBox("Song result\ncards", { color: C.blue, width: fixed(158), height: fixed(98), fontSize: 19 }),
      ]),
      grid(
        { name: "demo-controls-grid", width: fill, height: hug, columns: [fr(1), fr(1), fr(1)], columnGap: 12 },
        [
          metricCell("content weight", { fill: "#E4F1EC", line: C.accent, color: C.accent, bold: true }),
          metricCell("novelty", { fill: "#F4EFE7", line: C.orange, color: C.orange, bold: true }),
          metricCell("diversity", { fill: "#EEF3F7", line: C.blue, color: C.blue, bold: true }),
        ],
      ),
      row({ name: "metadata-display-row", width: fill, height: hug, gap: 10, align: "center" }, [
        diagramBox("KGRec\nitem ID", { color: C.accent, width: fixed(148), height: fixed(88), fontSize: 19 }),
        arrow("+ meta", 68),
        diagramBox("title / artist\ncover / link", { color: C.blue, width: fixed(170), height: fixed(88), fontSize: 19 }),
        arrow("->", 34),
        t("Ranking stays based on the research pipeline; enrichment only makes results readable during the live prototype demo.", {
          width: fixed(330),
          style: { fontSize: 22, bold: true, color: C.ink },
        }),
      ]),
    ]),
  ),
  note:
    "This is where the live prototype can be shown. Explain that the demo is not a fourth research model. It estimates a temporary user vector through fold-in, exposes product controls for content weight, novelty, and diversity, and uses NetEase only to make internal item IDs readable as real song cards.",
  source: "Product layer: SongRecDemo README; NetEase enrichment is display-side only",
});

addSlide({
  number: 13,
  title: "Final conclusion",
  subtitle: "The project became a full pipeline for understanding music recommendation from data to model to demo.",
  body: column({ name: "conclusion-left", width: fill, height: hug, gap: 22 }, [
    t("Main lesson", { style: { ...styles.label, color: C.accent } }),
    t("Recommendation systems are not only about maximizing accuracy; different algorithms create different discovery experiences.", {
      width: wrap(780),
      style: { fontSize: 37, bold: true, color: C.ink },
      name: "conclusion-claim",
    }),
  ]),
  visual: column({ name: "conclusion-right", width: fill, height: hug, gap: 16 }, [
    simpleFlow(["data", "models", "evaluation", "demo"], [C.blue, C.accent, C.orange, C.accent], { stepWidth: 154, fontSize: 20 }),
    bulletList(
      [
        "Popularity captures mainstream attention.",
        "Content-based tags push into niche areas but can become narrow.",
        "ALS learns hidden co-listening patterns and performed best overall.",
        "A real system also needs leakage control, cold-start handling, explanations, and metadata.",
      ],
      { name: "conclusion-bullets", style: styles.bodySmall, gap: 12 },
    ),
  ]),
  note:
    "Close by connecting the research and product work. Popularity is simple and safe, content-based recommendation can improve discovery but may be narrow, and ALS performed best because user-item behavior was the richest signal. The broader takeaway is that a recommender is a pipeline, not only a model.",
});

async function exportArtifacts() {
  for (const dir of [PREVIEW_DIR, LAYOUT_DIR]) {
    for (const file of fs.readdirSync(dir)) {
      fs.rmSync(path.join(dir, file), { recursive: true, force: true });
    }
  }

  const pptxBlob = await PresentationFile.exportPptx(presentation);
  await pptxBlob.save(PPTX_PATH);

  for (const [idx, slide] of presentation.slides.items.entries()) {
    const slideNo = String(idx + 1).padStart(2, "0");
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    fs.writeFileSync(path.join(PREVIEW_DIR, `slide-${slideNo}.png`), Buffer.from(await png.arrayBuffer()));
    const layout = await presentation.export({ slide, format: "layout" });
    fs.writeFileSync(path.join(LAYOUT_DIR, `slide-${slideNo}.layout.json`), Buffer.from(await layout.arrayBuffer()));
  }

  fs.writeFileSync(
    path.join(SCRATCH_DIR, "export-summary.json"),
    JSON.stringify(
      {
        deck: PPTX_PATH,
        slideCount: presentation.slides.items.length,
        previews: PREVIEW_DIR,
        layouts: LAYOUT_DIR,
        sources: [
          "artifacts/results/final_report.txt",
          "artifacts/results/comparison_table.md",
          "SongRecDemo/README.md",
          "docs/netease_api_setup.md",
        ],
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
}

await exportArtifacts();
console.log(`Exported ${PPTX_PATH}`);
