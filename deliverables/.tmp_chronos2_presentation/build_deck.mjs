import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  layers,
  shape,
  text,
} from "@oai/artifact-tool";

const OUT_DIR = "C:/Users/BQ6757/chronos2_v1/deliverables/.tmp_chronos2_presentation/output";
const FINAL_PPTX = "C:/Users/BQ6757/chronos2_v1/deliverables/Presentation_Chronos2_FR_2026-08-26.pptx";
const CSV_PATH = "C:/Users/BQ6757/chronos2_v1/runs/exports/2026-08-26/fr/autonomous/forecast_fr_2026-08-26_autonomous.csv";
const NOTE_PATH = "C:/Users/BQ6757/Downloads/Présentation courte — Forecast Fran.txt";
const REPORT_PATH = "C:/Users/BQ6757/chronos2_v1/runs/exports/2026-08-26/fr/autonomous/forecast_fr_2026-08-26_autonomous.html";

const W = 1280;
const H = 720;
const BLACK = "#000000";
const WHITE = "#FFFFFF";
const GREY = "#EDEDED";
const MID = "#B8BCC4";
const BLUE = "#3D8DFF";
const SKY = "#6DCBF4";
const PALE = "#D0EDFA";
const FONT = "Helvetica Neue";

function tx(value, left, top, width, height, fontSize, options = {}) {
  return text([value], {
    name: options.name,
    position: { left, top },
    width,
    height,
    style: {
      fontSize: `${fontSize}px`,
      typeface: FONT,
      color: options.color ?? BLACK,
      bold: options.bold ?? false,
      alignment: options.alignment ?? "left",
      verticalAlignment: options.verticalAlignment ?? "top",
      autoFit: options.autoFit ?? "shrinkText",
      wrap: "square",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  });
}

function rect(name, left, top, width, height, fill = GREY, radius = true, line = null) {
  return shape({
    name,
    geometry: radius ? "roundRect" : "rect",
    fill,
    line: line ?? { style: "solid", width: 0, fill },
    position: { left, top },
    width,
    height,
  });
}

function line(name, left, top, width, height = 0, fill = BLACK, strokeWidth = 1) {
  return shape({
    name,
    geometry: "straightConnector1",
    fill: "none",
    line: { style: "solid", width: strokeWidth, fill },
    position: { left, top },
    width: width === 0 ? 0.03 : width,
    height: height === 0 ? 0.03 : height,
  });
}

function circle(name, left, top, size, fill) {
  return shape({
    name,
    geometry: "ellipse",
    fill,
    line: { style: "solid", width: 0, fill },
    position: { left, top },
    width: size,
    height: size,
  });
}

function compose(slide, name, items) {
  slide.background.fill = WHITE;
  slide.compose(
    layers({ name, width: "fill", height: "fill" }, items),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 1 },
  );
}

function addFooter(items, n) {
  items.push(tx(String(n).padStart(2, "0"), 1184, 659, 55, 25, 13, {
    alignment: "right",
    verticalAlignment: "bottom",
    color: BLACK,
  }));
}

function notes(slide, body, sources) {
  const block = [
    body,
    "",
    "[Sources]",
    ...sources.map((s) => `- ${s}`),
    "[/Sources]",
  ].join("\n");
  slide.speakerNotes.textFrame.setText(block);
  slide.speakerNotes.setVisible(true);
}

function parseForecast(csv) {
  const lines = csv.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  const idx = Object.fromEntries(headers.map((h, i) => [h, i]));
  return lines.slice(1).map((row) => {
    const c = row.split(",");
    return {
      hour: Number(c[idx.local_hour]),
      q10: Number(c[idx.q10]),
      q50: Number(c[idx.q50]),
      q90: Number(c[idx.q90]),
    };
  });
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const forecast = parseForecast(await fs.readFile(CSV_PATH, "utf8"));
  const categories = forecast.map((d) => `${d.hour}h`);
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  // Slide 1 — sparse stacked-text cover, inspired by Codex Grid slide-01.
  {
    const slide = presentation.slides.add();
    const items = [
      rect("cover-accent", 41, 41, 10, 68, BLUE, false),
      tx("PRÉVISION DAY-AHEAD  •  FRANCE", 70, 47, 560, 42, 24, { bold: true }),
      tx("Chronos-2", 41, 190, 995, 90, 76, { bold: true }),
      tx("Prévoir demain,\nheure par heure", 41, 295, 1030, 170, 70, { bold: true }),
      tx("Fonctionnement du modèle  •  Résultats du 26 août 2026", 41, 520, 880, 48, 27),
      line("cover-rule", 41, 610, 1198, 0, MID, 1),
      tx("Courbe centrale P50 + scénarios P10 / P90", 41, 631, 650, 28, 17, { color: "#4D5158" }),
    ];
    compose(slide, "chronos-cover", items);
    notes(slide,
      "Présenter l’objectif en une phrase : prévoir le prix day-ahead français pour chacune des 24 heures du lendemain, avec une courbe centrale et une plage d’incertitude.",
      [NOTE_PATH],
    );
  }

  // Slide 2 — process timeline, inspired by Codex Grid slide-17.
  {
    const slide = presentation.slides.add();
    const items = [
      tx("Du signal brut à une prévision contrôlée", 41, 36, 1197, 66, 39, { bold: true }),
      tx("Une chaîne causale en cinq étapes", 41, 105, 680, 35, 21, { color: "#4D5158" }),
      line("pipeline-line", 78, 307, 1093, 0, BLACK, 2),
    ];
    const steps = [
      { x: 78, c: BLACK, label: "D−1", title: "Entrées", body: "Prix récents\nCalendrier\nCharges résiduelles\nFR, DE, BE, NL, ES" },
      { x: 309, c: BLUE, label: "BASE", title: "Chronos-2", body: "Construit la forme\nde la journée et\nproduit P10, P50, P90" },
      { x: 540, c: SKY, label: "AJUST.", title: "Correcteur", body: "Anticipe l’erreur\nrécurrente et déplace\nles trois quantiles" },
      { x: 771, c: BLACK, label: "GEL", title: "Contrôles", body: "24 heures\nP10 ≤ P50 ≤ P90\nAbsence de fuite future" },
      { x: 1002, c: BLACK, label: "APRÈS", title: "Évaluation", body: "Prix réels, métriques\net comparaison à Storm\nuniquement après coup" },
    ];
    for (const [i, s] of steps.entries()) {
      items.push(tx(s.label, s.x, 225, 180, 30, 17, { bold: true, color: i === 2 ? BLUE : BLACK }));
      items.push(circle(`node-${i}`, s.x, 294, 26, s.c));
      items.push(tx(s.title, s.x, 354, 190, 38, 25, { bold: true }));
      items.push(tx(s.body, s.x, 404, 182, 125, 17, { color: "#33363B" }));
    }
    items.push(rect("causal-callout", 41, 579, 1198, 58, "#F2F2F2", true));
    items.push(tx("Principe clé : seules les informations disponibles au moment du forecast entrent dans la chaîne.", 68, 595, 1145, 28, 20, { bold: true }));
    addFooter(items, 2);
    compose(slide, "chronos-pipeline", items);
    notes(slide,
      "Dérouler la chaîne de gauche à droite. Insister sur la causalité : aucune information connue après l’heure du forecast n’est utilisée. Storm est un benchmark ex-post, jamais une variable d’entrée.",
      [NOTE_PATH],
    );
  }

  // Slide 3 — two-column comparison, inspired by Codex Grid slide-11.
  {
    const slide = presentation.slides.add();
    const items = [
      tx("Deux étages, deux rôles complémentaires", 41, 36, 1197, 66, 39, { bold: true }),
      tx("Chronos-2 construit la prévision de base ; le correcteur apprend quand cette base se trompe.", 41, 110, 1197, 38, 21, { color: "#4D5158" }),
      rect("chronos-card", 41, 181, 581, 431, "#F2F2F2", true),
      rect("corrector-card", 657, 181, 581, 431, "#F2F2F2", true),
      circle("stage-one", 72, 211, 36, BLUE),
      tx("1", 82, 217, 16, 22, 17, { bold: true, color: WHITE, alignment: "center" }),
      tx("Chronos-2", 125, 207, 400, 48, 32, { bold: true }),
      tx("Modèle pré-entraîné de séries temporelles", 72, 271, 500, 34, 20, { bold: true, color: BLUE }),
      tx("• Jusqu’à 2 048 heures de prix récent\n• Calendrier + contexte électrique des 5 pays\n• Reconnaît cycles, pics et changements de régime\n• Produit une distribution, pas un point unique", 72, 324, 495, 142, 19),
      line("quantile-rule", 72, 492, 494, 0, MID, 1),
      tx("P10", 78, 519, 78, 28, 19, { bold: true, color: "#479AC0" }),
      tx("P50", 255, 519, 78, 28, 19, { bold: true, color: BLUE }),
      tx("P90", 432, 519, 78, 28, 19, { bold: true, color: "#479AC0" }),
      tx("scénario bas", 78, 551, 110, 25, 15, { color: "#55585E" }),
      tx("central", 255, 551, 110, 25, 15, { color: "#55585E" }),
      tx("scénario haut", 432, 551, 120, 25, 15, { color: "#55585E" }),
      circle("stage-two", 688, 211, 36, SKY),
      tx("2", 698, 217, 16, 22, 17, { bold: true, color: BLACK, alignment: "center" }),
      tx("Correcteur résiduel", 741, 207, 440, 48, 32, { bold: true }),
      tx("Cible : prix réel − P50 Chronos-2", 688, 271, 500, 34, 20, { bold: true, color: BLUE }),
      tx("CatBoost et HistGradientBoosting ajoutent des arbres successifs : chaque arbre corrige une partie de l’erreur laissée par les précédents.", 688, 320, 500, 84, 19),
      rect("blend", 688, 423, 500, 61, WHITE, true),
      tx("CatBoost  50 %   +   HistGradientBoosting  50 %", 710, 441, 457, 25, 18, { bold: true, alignment: "center" }),
      tx("Apprentissage causal out-of-fold  •  correction limitée à ±40 €/MWh", 688, 506, 500, 48, 18),
      tx("Exemple   150 + 8 = 158 €/MWh", 688, 564, 500, 31, 22, { bold: true, color: BLUE }),
    ];
    addFooter(items, 3);
    compose(slide, "chronos-two-stages", items);
    notes(slide,
      "Chronos-2 apporte une base probabiliste générale. Le correcteur ne re-prédit pas directement le prix : il prédit le résidu. Les deux algorithmes sont des ensembles d’arbres construits séquentiellement. CatBoost capte bien les interactions complexes ; HistGradientBoosting regroupe les valeurs en intervalles pour apprendre efficacement. Leurs sorties sont moyennées. À chaque forecast, les modèles sont recréés sur le même historique figé : il s’agit d’un réentraînement technique, pas d’un recalibrage avec les toutes dernières observations. Le même déplacement est ajouté à P10, P50 et P90, donc la largeur de la bande reste inchangée.",
      [NOTE_PATH],
    );
  }

  // Slide 4 — line chart with stat callouts, inspired by Codex Grid slide-21.
  {
    const slide = presentation.slides.add();
    const items = [
      tx("Forecast France — 26 août 2026", 41, 36, 1197, 66, 39, { bold: true }),
      tx("P50 dessine la trajectoire centrale ; P10 et P90 matérialisent l’incertitude horaire.", 41, 105, 1197, 35, 21, { color: "#4D5158" }),
      rect("avg-card", 823, 182, 190, 174, "#F2F2F2", true),
      rect("range-card", 1041, 182, 198, 174, "#F2F2F2", true),
      tx("144,0", 849, 213, 140, 58, 43, { bold: true, color: BLUE, alignment: "center" }),
      tx("€/MWh\nP50 moyenne", 849, 284, 140, 49, 18, { alignment: "center" }),
      tx("75,9 → 191,5", 1058, 220, 164, 49, 28, { bold: true, alignment: "center" }),
      tx("€/MWh\nde 13 h à 20 h", 1065, 286, 150, 48, 18, { alignment: "center" }),
      tx("Lecture rapide", 823, 403, 416, 32, 23, { bold: true }),
      tx("• Creux à 13 h\n• Pic du soir à 20 h\n• Bande la plus large à 12 h : 138,7 €/MWh\n• L’incertitude varie fortement selon l’heure", 823, 453, 407, 135, 19),
      line("stats-rule", 823, 615, 416, 0, MID, 1),
      tx("Les quantiles sont des scénarios, pas des bornes garanties.", 823, 630, 395, 29, 16, { color: "#55585E" }),
    ];
    addFooter(items, 4);
    compose(slide, "chronos-forecast-chart", items);
    slide.charts.add("line", {
      position: { left: 41, top: 159, width: 739, height: 485 },
      categories,
      series: [
        {
          name: "P10",
          values: forecast.map((d) => d.q10),
          line: { style: "solid", width: 2, fill: SKY },
          marker: { symbol: "none" },
        },
        {
          name: "P50",
          values: forecast.map((d) => d.q50),
          line: { style: "solid", width: 4, fill: BLUE },
          marker: { symbol: "circle", size: 4 },
        },
        {
          name: "P90",
          values: forecast.map((d) => d.q90),
          line: { style: "solid", width: 2, fill: "#8CCBE8" },
          marker: { symbol: "none" },
        },
      ],
      hasLegend: true,
      legend: {
        position: "bottom",
        overlay: false,
        textStyle: { fontSize: 13, fill: BLACK },
      },
      dataLabels: { showValue: false },
      chartFill: WHITE,
      chartLine: { style: "solid", width: 0, fill: WHITE },
      plotAreaFill: { type: "none" },
      plotAreaLine: { style: "solid", width: 0, fill: WHITE },
      xAxis: {
        visible: true,
        line: { style: "solid", width: 1, fill: MID },
        textStyle: { fontSize: 11, fill: BLACK },
      },
      yAxis: {
        visible: true,
        title: { text: "€/MWh", textStyle: { fontSize: 12, fill: BLACK } },
        min: 0,
        max: 260,
        majorUnit: 50,
        majorGridlines: { style: "solid", width: 1, fill: GREY },
        line: { style: "solid", width: 0, fill: WHITE },
        textStyle: { fontSize: 11, fill: BLACK },
      },
      lineOptions: { grouping: "standard", smooth: false },
    });
    notes(slide,
      "Commencer par la forme de la journée : baisse jusqu’au milieu de journée puis nette remontée en soirée. P50 vaut en moyenne 144,0 €/MWh, avec un minimum de 75,9 €/MWh à 13 h et un maximum de 191,5 €/MWh à 20 h. La bande P10–P90 est particulièrement large à 12 h : l’incertitude n’est donc pas uniforme.",
      [CSV_PATH, NOTE_PATH],
    );
  }

  // Slide 5 — sparse evidence and takeaway.
  {
    const slide = presentation.slides.add();
    const items = [
      tx("Ce que montrent les résultats historiques", 41, 36, 1197, 66, 39, { bold: true }),
      tx("Le correcteur améliore Chronos-2 brut ; Storm reste légèrement devant sur la fenêtre complète.", 41, 105, 1197, 35, 21, { color: "#4D5158" }),
      tx("MAE  •  plus bas = meilleur", 41, 184, 500, 31, 18, { bold: true }),
      tx("Chronos-2", 41, 247, 165, 28, 19, { bold: true }),
      rect("bar-chronos", 220, 245, 430, 32, "#D7D9DE", true),
      tx("13,11", 665, 244, 82, 31, 20, { bold: true, alignment: "right" }),
      tx("Corrigé", 41, 319, 165, 28, 19, { bold: true, color: BLUE }),
      rect("bar-corrected", 220, 317, 408, 32, BLUE, true),
      tx("12,44", 665, 316, 82, 31, 20, { bold: true, alignment: "right", color: BLUE }),
      tx("Storm", 41, 391, 165, 28, 19, { bold: true }),
      rect("bar-storm", 220, 389, 383, 32, BLACK, true),
      tx("11,67", 665, 388, 82, 31, 20, { bold: true, alignment: "right" }),
      line("evidence-separator", 790, 184, 0, 321, MID, 1),
      tx("−5,1 %", 841, 210, 355, 82, 58, { bold: true, color: BLUE }),
      tx("de MAE par rapport à Chronos-2 brut", 841, 299, 355, 56, 23, { bold: true }),
      tx("Corrélation : 0,93", 841, 382, 355, 34, 22, { bold: true }),
      tx("Win rate face à Storm : 47,56 %", 841, 434, 355, 34, 22),
      rect("storm-note", 841, 493, 355, 66, "#F2F2F2", true),
      tx("Storm = benchmark après coup,\njamais une entrée du modèle", 864, 508, 310, 41, 18, { bold: true }),
      rect("takeaway", 41, 595, 1198, 70, PALE, true),
      tx("À retenir : lire P50 avec P10–P90 comme une estimation encadrée — pas comme une valeur certaine.", 70, 615, 1138, 31, 21, { bold: true }),
    ];
    addFooter(items, 5);
    compose(slide, "chronos-evidence", items);
    notes(slide,
      "Le correcteur réduit la MAE de Chronos-2 d’environ 5 %, à 12,44 €/MWh, avec une corrélation de 0,93. Storm conserve une MAE plus basse, à 11,67 €/MWh, et le modèle gagne 47,56 % des heures face à lui. Le message final est donc équilibré : gain réel par rapport à la base, mais marge d’amélioration encore visible face au benchmark.",
      [REPORT_PATH, NOTE_PATH],
    );
  }

  for (const [i, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    await writeBlob(path.join(OUT_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(OUT_DIR, `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(OUT_DIR, "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(`Wrote ${FINAL_PPTX}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
