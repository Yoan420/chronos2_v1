import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const WORKSPACE_DIR = process.env.WORKSPACE_DIR;
const TMP_DIR = process.env.TMP_DIR;
const SKILL_DIR = process.env.SKILL_DIR;
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON;
const FINAL_PPTX = process.env.FINAL_PPTX;
const REFERENCE_PPTX = process.env.REFERENCE_PPTX;
const FORECAST_CSV = process.env.FORECAST_CSV;

for (const [name, value] of Object.entries({
  WORKSPACE_DIR,
  TMP_DIR,
  SKILL_DIR,
  RUNTIME_PYTHON,
  FINAL_PPTX,
  REFERENCE_PPTX,
  FORECAST_CSV,
})) {
  if (!value) throw new Error(`Missing environment variable: ${name}`);
}

const FONT = "Helvetica Neue";
const NAVY = "#08284F";
const BLUE = "#175CD4";
const MID_BLUE = "#0B3668";
const PALE_BLUE = "#EDF5FE";
const SLATE = "#7E91A8";
const WHITE = "#FFFFFF";
const BODY = "#243B57";
const GRID = "#D9E4F0";

const presentation = await PresentationFile.importPptx(await FileBlob.load(REFERENCE_PPTX));

const chosenOriginalNumbers = [1, 20, 5, 24, 23, 31];
const chosenSlides = chosenOriginalNumbers.map((number) => presentation.slides.getItem(number - 1));
const chosenIds = new Set(chosenSlides.map((slide) => slide.id));

for (const slide of [...presentation.slides.items].reverse()) {
  if (!chosenIds.has(slide.id)) slide.delete();
}
for (let index = 0; index < chosenSlides.length; index += 1) {
  chosenSlides[index].moveTo(index);
}

const [cover, pipeline, stages, forecast, performance, takeaway] = chosenSlides;

function setPlain(shape, text, style = {}) {
  shape.text.set(text);
  shape.text.style = {
    typeface: FONT,
    fill: BODY,
    autoFit: "shrinkText",
    wrap: "square",
    ...style,
  };
}

function setTitle(shape, text, extra = {}) {
  setPlain(shape, text, {
    fontSize: 48,
    fill: NAVY,
    lineSpacing: 0.94,
    ...extra,
  });
}

function setRich(shape, paragraphs, style = {}) {
  shape.text.set(paragraphs);
  shape.text.style = {
    typeface: FONT,
    fill: BODY,
    autoFit: "shrinkText",
    wrap: "square",
    verticalAlignment: "top",
    ...style,
  };
}

function setNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines);
  slide.speakerNotes.setVisible(true);
}

function updateSlideNumber(slide, number) {
  for (const shape of slide.shapes.items) {
    if (shape.placeholderType === "slideNumber") setPlain(shape, String(number), { fontSize: 15, fill: SLATE });
  }
}

// Slide 1: cover.
setPlain(cover.shapes.getItemAt(0), "Prévision day-ahead France\nChronos-2 et ses correcteurs", {
  fontSize: 78,
  fill: WHITE,
  lineSpacing: 0.9,
});
setPlain(cover.shapes.getItemAt(1), "Prévision du 9 septembre 2026\nRapport France", {
  fontSize: 24,
  fill: WHITE,
});
setPlain(cover.shapes.getItemAt(2), "Présentation simple\nModèle et résultats", {
  fontSize: 24,
  fill: WHITE,
});
setPlain(cover.shapes.getItemAt(3), "09 SEPT. 2026", {
  fontSize: 22,
  fill: WHITE,
  alignment: "right",
});
setPlain(cover.shapes.getItemAt(4), "FORECAST FRANCE", {
  fontSize: 22,
  bold: true,
  fill: WHITE,
});
setNotes(cover, [
  "Objectif oral : présenter en quelques minutes le fonctionnement du modèle et les principaux résultats du rapport France.",
  `Sources : ${FORECAST_CSV}`,
  "Rapport associé : runs/exports/2026-09-09/fr/kalman/forecast_fr_2026-09-09_kalman.html",
]);

// Slide 2: pipeline.
setTitle(pipeline.shapes.getItemAt(1), "Du signal brut au rapport publié");
const timelineLine = pipeline.shapes.getItemAt(4);
timelineLine.position = { ...timelineLine.position, width: 1203.2 };
const pipelineBodies = [
  ["Données disponibles à D−1", "Prix récents, calendrier et prévisions de charge résiduelle sur cinq pays."],
  ["Prévision et correction", "Chronos-2 produit P10, P50 et P90. Le correcteur résiduel apprend les erreurs récurrentes."],
  ["Kalman et publication", "Le filtre admissible le plus performant est retenu, puis les contrôles produisent le CSV et le rapport HTML."],
];
for (let i = 0; i < 3; i += 1) {
  setRich(pipeline.shapes.getItemAt(i === 0 ? 0 : i + 1), [
    [{ run: pipelineBodies[i][0], textStyle: { bold: true, fontSize: "18pt", typeface: FONT, color: NAVY } }],
    [{ run: pipelineBodies[i][1], textStyle: { fontSize: "16pt", typeface: FONT, color: BODY } }],
  ], { lineSpacing: 1.05 });
}
for (const [shapeIndex, label] of [[8, "1 · ENTRÉES"], [9, "2 · MODÈLE"], [10, "3 · SORTIE"]]) {
  setPlain(pipeline.shapes.getItemAt(shapeIndex), label, { fontSize: 17, bold: true, fill: BLUE });
}
updateSlideNumber(pipeline, 2);
setNotes(pipeline, [
  "Message clé : le modèle est un pipeline causal. Au moment de prévoir une journée, seules les informations déjà disponibles sont utilisées.",
  "L’absence de fuite future signifie qu’aucune observation postérieure à la date de prévision n’entre dans l’entraînement, la sélection ou la correction appliquée à cette prévision.",
  "Sources : Forecast.ps1, chronos2_hourly/kalman_residual.py, deliverables/Note_explicative_Kalman.txt.",
]);

// Slide 3: model stages.
setTitle(stages.shapes.getItemAt(3), "Trois étages, trois rôles complémentaires");
setPlain(stages.shapes.getItemAt(7), "FONCTIONNEMENT DU MODÈLE", { fontSize: 17, bold: true, fill: BLUE });
setPlain(stages.shapes.getItemAt(5), "Chaque étage traite une source d’erreur différente, du signal de fond jusqu’aux écarts les plus récents.", {
  fontSize: 24,
  fill: BODY,
});
const stageContent = [
  ["Chronos-2", [
    "Transformer préentraîné de type encodeur-only",
    "Lit l’historique et les variables connues",
    "Produit directement P10, P50 et P90",
  ]],
  ["Correcteur résiduel", [
    "Moyenne de CatBoost et HistGradientBoosting",
    "Prévoit l’erreur restante de Chronos-2",
    "Décale les trois quantiles de façon identique",
  ]],
  ["Filtre de Kalman", [
    "Suit le biais récent au fil des jours",
    "Teste plusieurs dynamiques sans fuite future",
    "Peut conserver la prévision inchangée",
  ]],
];
for (let i = 0; i < 3; i += 1) {
  const [heading, bullets] = stageContent[i];
  setRich(stages.shapes.getItemAt(i), [
    [{ run: heading, textStyle: { bold: true, fontSize: "21pt", typeface: FONT, color: NAVY } }],
    ...bullets.map((text) => ({
      bulletCharacter: "•",
      marginLeft: 18 * 12700,
      indent: -10 * 12700,
      runs: [{ run: text, textStyle: { fontSize: "16.5pt", typeface: FONT, color: BODY } }],
    })),
  ], { lineSpacing: 1.02, insets: { top: 6, right: 12, bottom: 6, left: 12 } });
}
updateSlideNumber(stages, 3);
setNotes(stages, [
  "Chronos-2 est un transformer encodeur-only pour séries temporelles. Il encode le contexte historique et les covariables, puis une tête de prédiction produit l’horizon futur sous forme de quantiles.",
  "Le correcteur résiduel apprend la quantité observée moins la médiane Chronos-2. CatBoost et HistGradientBoosting construisent chacun une suite d’arbres qui corrige progressivement les erreurs du modèle précédent. Leurs prédictions sont moyennées à parts égales.",
  "Le filtre de Kalman assimile ensuite uniquement les erreurs déjà observées. La gouvernance compare plusieurs filtres à une identité sans correction et n’applique un filtre que s’il apporte un gain récent.",
  "Sources : https://huggingface.co/amazon/chronos-2 ; https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/model.py ; chronos2_hourly/kalman_residual.py ; deliverables/Note_explicative_Kalman.txt.",
]);

// Parse the 24-hour forecast used on slide 4.
const csvText = await fs.readFile(FORECAST_CSV, "utf8");
const csvLines = csvText.trim().split(/\r?\n/);
const headers = csvLines[0].split(",");
const col = Object.fromEntries(headers.map((name, index) => [name, index]));
const rows = csvLines.slice(1).map((line) => {
  const values = line.split(",");
  return {
    hour: Number(values[col.local_hour]),
    q10: Number(values[col.q10]),
    q50: Number(values[col.q50]),
    q90: Number(values[col.q90]),
  };
});
const hourLabels = rows.map(({ hour }) => `${String(hour).padStart(2, "0")} h`);

// Slide 4: forecast chart.
setTitle(forecast.shapes.getItemAt(0), "La prévision du 9 septembre 2026");
setPlain(forecast.shapes.getItemAt(7), "Le creux intervient à 13 h, puis la courbe remonte rapidement vers un pic à 19 h.", {
  fontSize: 25,
  fill: BODY,
});
setPlain(forecast.shapes.getItemAt(2), "152,8", { fontSize: 66, fill: BLUE, bold: true });
setPlain(forecast.shapes.getItemAt(3), "Prix moyen P50\nEUR/MWh", { fontSize: 20, fill: BODY });
setPlain(forecast.shapes.getItemAt(5), "68,7", { fontSize: 66, fill: NAVY, bold: true });
setPlain(forecast.shapes.getItemAt(6), "Largeur maximale P10–P90\nà 16 h", { fontSize: 20, fill: BODY });
const importedForecastChart = forecast.charts.items[0];
const forecastChartPosition = importedForecastChart.position;
forecast.charts.deleteById(importedForecastChart.id);
forecast.charts.add("line", {
  position: forecastChartPosition,
  title: "Prix horaire prévu, EUR/MWh",
  titlePlacement: "aboveChart",
  titleTextStyle: { typeface: FONT, fontSize: 18, fill: NAVY, bold: true },
  categories: hourLabels,
  series: [
    { name: "P10", values: rows.map((row) => row.q10), line: { style: "solid", fill: "#84B9F3", width: 2 }, marker: { symbol: "none" } },
    { name: "P90", values: rows.map((row) => row.q90), line: { style: "solid", fill: "#4D8FE7", width: 2 }, marker: { symbol: "none" } },
    { name: "P50", values: rows.map((row) => row.q50), line: { style: "solid", fill: NAVY, width: 4 }, marker: { symbol: "none" } },
  ],
  hasLegend: true,
  legend: { position: "bottom", overlay: false, textStyle: { typeface: FONT, fontSize: 14, fill: BODY } },
  lineOptions: { grouping: "standard", smooth: false, varyColors: false },
  xAxis: {
    visible: true,
    textStyle: { typeface: FONT, fontSize: 10, fill: BODY },
    line: { style: "solid", fill: GRID, width: 1 },
    majorGridlines: null,
  },
  yAxis: {
    visible: true,
    min: 40,
    max: 260,
    majorUnit: 40,
    numberFormatCode: "0",
    textStyle: { typeface: FONT, fontSize: 12, fill: BODY },
    line: { style: "solid", fill: GRID, width: 1 },
    majorGridlines: { style: "solid", fill: GRID, width: 1 },
  },
  chartFill: { type: "solid", color: WHITE },
  plotAreaFill: { type: "solid", color: WHITE },
});
updateSlideNumber(forecast, 4);
setNotes(forecast, [
  "P50 est la trajectoire centrale. P10 et P90 encadrent l’incertitude du modèle : la valeur réalisée est attendue entre ces deux bornes dans environ 80 % des cas si le modèle est bien calibré.",
  "Chiffres du jour : moyenne P50 152,81 EUR/MWh ; minimum 78,60 EUR/MWh à 13 h ; maximum 222,89 EUR/MWh à 19 h ; largeur maximale P10–P90 68,65 EUR/MWh à 16 h.",
  `Source : ${FORECAST_CSV}`,
]);

// Slide 5: historical performance chart.
setTitle(performance.shapes.getItemAt(0), "Le Kalman améliore la précision moyenne");
setPlain(performance.shapes.getItemAt(6), "Évaluation causale sur 365 jours et 8 760 heures. La correction de Kalman réduit la MAE et la RMSE, tout en rapprochant le biais de zéro. La MAE des variations horaires est légèrement moins bonne.", {
  fontSize: 24,
  fill: BODY,
});
setPlain(performance.shapes.getItemAt(1), "1,2 %", { fontSize: 66, fill: BLUE, bold: true });
setPlain(performance.shapes.getItemAt(2), "Gain de MAE\npar rapport au correcteur résiduel", { fontSize: 20, fill: BODY });
setPlain(performance.shapes.getItemAt(3), "−0,56", { fontSize: 66, fill: NAVY, bold: true });
setPlain(performance.shapes.getItemAt(4), "Biais moyen après Kalman\nEUR/MWh", { fontSize: 20, fill: BODY });
const importedPerfChart = performance.charts.items[0];
const perfChartPosition = importedPerfChart.position;
performance.charts.deleteById(importedPerfChart.id);
performance.charts.add("bar", {
  position: perfChartPosition,
  title: "Erreur historique, EUR/MWh",
  titlePlacement: "aboveChart",
  titleTextStyle: { typeface: FONT, fontSize: 18, fill: NAVY, bold: true },
  categories: ["MAE", "RMSE"],
  series: [
    { name: "Correcteur résiduel", values: [12.7050, 20.3742], fill: SLATE, line: { style: "solid", fill: SLATE, width: 1 } },
    { name: "Après Kalman", values: [12.5521, 20.0488], fill: BLUE, line: { style: "solid", fill: BLUE, width: 1 } },
  ],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 70, varyColors: false },
  hasLegend: true,
  legend: { position: "bottom", overlay: false, textStyle: { typeface: FONT, fontSize: 13, fill: BODY } },
  dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: FONT, fontSize: 13, fill: NAVY, bold: true } },
  xAxis: {
    visible: true,
    textStyle: { typeface: FONT, fontSize: 14, fill: BODY, bold: true },
    line: { style: "solid", fill: GRID, width: 1 },
    majorGridlines: null,
  },
  yAxis: {
    visible: true,
    min: 0,
    max: 24,
    majorUnit: 4,
    numberFormatCode: "0.0",
    textStyle: { typeface: FONT, fontSize: 12, fill: BODY },
    line: { style: "solid", fill: GRID, width: 1 },
    majorGridlines: { style: "solid", fill: GRID, width: 1 },
  },
  chartFill: { type: "solid", color: PALE_BLUE },
  plotAreaFill: { type: "solid", color: PALE_BLUE },
});
updateSlideNumber(performance, 5);
setNotes(performance, [
  "La MAE est la taille moyenne des erreurs, sans tenir compte du signe. Le biais est la moyenne signée : un biais proche de zéro indique qu’il n’y a pas de surestimation ou de sous-estimation systématique.",
  "Résultats : MAE 12,5521 contre 12,7050, gain 1,2034 % ; RMSE 20,0488 contre 20,3742, gain 1,5974 % ; biais −0,5590 contre −1,2297 ; corrélation 0,9320 contre 0,9299.",
  "Point de vigilance : ramp MAE 7,2255 contre 7,2052. Le Kalman améliore le niveau de prix moyen, mais pas toutes les transitions horaires.",
  "Source : runs/exports/2026-09-09/fr/kalman/forecast_fr_2026-09-09_kalman.html et kalman_filter_audit.json.",
]);

// Slide 6: takeaway.
setPlain(takeaway.shapes.getItemAt(0), "À retenir\nUne prévision, une fourchette, des garde-fous", {
  fontSize: 66,
  fill: WHITE,
  lineSpacing: 0.9,
});
setPlain(takeaway.shapes.getItemAt(1), "Présenter ensemble la courbe P50,\nl’intervalle P10–P90 et la performance historique.", {
  fontSize: 24,
  fill: WHITE,
});
setPlain(takeaway.shapes.getItemAt(2), "FORECAST FRANCE · 9 SEPTEMBRE 2026", {
  fontSize: 22,
  bold: true,
  fill: WHITE,
});
setNotes(takeaway, [
  "Conclusion proposée : Chronos-2 fournit la structure principale, le correcteur résiduel retire les erreurs récurrentes et le Kalman adapte prudemment la prévision au régime récent.",
  "Pour lire le rapport, commencer par P50, regarder ensuite la largeur P10–P90, puis mettre la prévision en perspective avec les métriques historiques.",
  "Le benchmark Storm sert à comparer les résultats après coup. Il n’entre pas dans le filtre de Kalman.",
  "Source : runs/exports/2026-09-09/fr/kalman/forecast_fr_2026-09-09_kalman.html et kalman_filter_audit.json.",
]);

const draftPath = path.join(TMP_DIR, "draft_model_report.pptx");
await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
await (await PresentationFile.exportPptx(presentation)).save(draftPath);

const requirements = {
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [4, 5],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  nativeChartTargetApplication: "powerpoint",
  materializeLiteralChartWorkbooks: true,
  sourceTemplatePath: REFERENCE_PPTX,
  requiredTemplateReferenceSlides: [1, 5, 20, 23, 24, 31],
  minimumTemplateCoverageRatio: 0,
  requireExactTemplateDimensions: true,
  requireTemplatePlaceholderGeometry: true,
};
const fontPolicy = {
  basis: "reference",
  families: [FONT],
  referencePath: REFERENCE_PPTX,
  referenceSha256: "ec2084d143a4d52857cc06c24129abbb45c45d04b90cca06e319ac26d3cadd4f",
};
const expectedSlideSizeEmu = "12192000,6858000";
const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"),
).href);
const stagingDir = path.join(TMP_DIR, "finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const result = await finalizePresentation({
  ...requirements,
  workspaceDir: WORKSPACE_DIR,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});

console.log(JSON.stringify({ draftPath, finalPath: result.finalPath ?? FINAL_PPTX, result }, null, 2));
