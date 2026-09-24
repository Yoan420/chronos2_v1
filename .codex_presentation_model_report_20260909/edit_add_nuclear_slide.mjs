import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const WORKSPACE_DIR = process.env.WORKSPACE_DIR;
const TMP_DIR = process.env.TMP_DIR;
const SKILL_DIR = process.env.SKILL_DIR;
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON;
const SOURCE_PPTX = process.env.SOURCE_PPTX;
const FINAL_PPTX = process.env.FINAL_PPTX;

for (const [name, value] of Object.entries({
  WORKSPACE_DIR,
  TMP_DIR,
  SKILL_DIR,
  RUNTIME_PYTHON,
  SOURCE_PPTX,
  FINAL_PPTX,
})) {
  if (!value) throw new Error(`Missing environment variable: ${name}`);
}

const FONT = "Helvetica Neue";
const NAVY = "#08284F";
const BLUE = "#175CD4";
const SLATE = "#7E91A8";
const BODY = "#243B57";

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
    if (shape.placeholderType === "slideNumber") {
      setPlain(shape, String(number), { fontSize: 15, fill: SLATE });
    }
  }
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE_PPTX));
if (presentation.slides.count !== 6) {
  throw new Error(`Expected 6 source slides, found ${presentation.slides.count}`);
}

// Duplicate the existing three-column explanation slide and insert it immediately after.
const nuclear = presentation.slides.getItem(2).duplicate();
nuclear.moveTo(3);

setTitle(nuclear.shapes.getItemAt(3), "Introduction de la génération nucléaire française");
setPlain(nuclear.shapes.getItemAt(7), "CHALLENGER NUCLÉAIRE", {
  fontSize: 17,
  bold: true,
  fill: BLUE,
});
setPlain(
  nuclear.shapes.getItemAt(5),
  "Le pipeline ajoute une prévision de production, jamais la réalisation. Cette variante reste séparée du modèle standard.",
  { fontSize: 24, fill: BODY },
);

const nuclearContent = [
  ["Source et gel", [
    "Prévision Saturn de production nucléaire FR en GW",
    "Vintage disponible à D−1, 08 h, heure de Paris",
    "730 jours historiques et la journée à prévoir",
  ]],
  ["Dans Chronos-2", [
    "Ajoutée au contexte historique",
    "Ajoutée comme covariable future connue",
    "Chronos-2 reste gelé, sans nouvel entraînement",
  ]],
  ["Dans les correcteurs", [
    "Feature dédiée du correcteur résiduel",
    "Variable « market » supplémentaire dans Kalman",
    "Charge résiduelle inchangée et unités séparées",
  ]],
];

for (let index = 0; index < nuclearContent.length; index += 1) {
  const [heading, bullets] = nuclearContent[index];
  setRich(nuclear.shapes.getItemAt(index), [
    [{ run: heading, textStyle: { bold: true, fontSize: "21pt", typeface: FONT, color: NAVY } }],
    ...bullets.map((text) => ({
      bulletCharacter: "•",
      marginLeft: 18 * 12700,
      indent: -10 * 12700,
      runs: [{ run: text, textStyle: { fontSize: "16.5pt", typeface: FONT, color: BODY } }],
    })),
  ], {
    lineSpacing: 1.02,
    insets: { top: 6, right: 12, bottom: 6, left: 12 },
  });
}

setNotes(nuclear, [
  "La source est la série Saturn power.fr.generation.nuclear.gw.fcst. Elle représente une prévision de production nucléaire française en GW, distincte de la disponibilité REMIT et de la production réalisée.",
  "Pour chaque journée de livraison, le pipeline sélectionne le dernier vintage disponible à D−1 08 h, heure de Paris. Il utilise 730 jours historiques complets ainsi que la journée à prévoir.",
  "La variable nucléaire entre dans le contexte de Chronos-2 et dans ses covariables futures connues. Le réseau Chronos-2 reste gelé : il n’est ni réentraîné ni adapté par LoRA.",
  "Le correcteur résiduel reçoit une feature dédiée nommée known_fr_nuclear_generation_fcst_gw_oracle. Le mot oracle est un nom technique du mécanisme de covariable future connue : les valeurs restent des prévisions disponibles à la date de coupure, jamais des réalisations futures.",
  "Le Kalman ajoute le nucléaire au groupe market. La charge résiduelle n’est pas modifiée, afin de ne pas mélanger une production en GW avec un agrégat de charge résiduelle.",
  "Cette fonctionnalité est un challenger séparé. Elle produit nuclear_autonomous ou nuclear_kalman et ne modifie pas le pipeline standard. Son gain propre doit encore être isolé par une comparaison contrôlée avec et sans nucléaire.",
  "Sources : NUCLEAR_FORECAST.md ; chronos2_hourly/nuclear_forecast.py ; config/nuclear_forecast.yaml ; Forecast.ps1.",
]);

for (let index = 0; index < presentation.slides.count; index += 1) {
  updateSlideNumber(presentation.slides.getItem(index), index + 1);
}

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const candidatePath = path.join(TMP_DIR, "candidate_nuclear.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 7,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [5, 6],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  nativeChartTargetApplication: "powerpoint",
  materializeLiteralChartWorkbooks: false,
};
const fontPolicy = {
  basis: "reference",
  families: [FONT],
  referencePath: SOURCE_PPTX,
  referenceSha256: "e275eb763bd5264198fa78928900eedcb9ff854f330f4737a337ff022656a587",
};
const expectedSlideSizeEmu = "12192000,6858000";
const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"),
).href);
const stagingDir = path.join(TMP_DIR, "finalizer_nuclear");
await fs.mkdir(stagingDir, { recursive: true });
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

console.log(JSON.stringify({ finalPath: result.finalPath ?? FINAL_PPTX, result }, null, 2));
