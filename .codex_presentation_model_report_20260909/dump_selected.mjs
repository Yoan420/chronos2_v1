import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const presentation = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));

function safe(value) {
  try { return JSON.parse(JSON.stringify(value)); } catch { return String(value); }
}

for (const slideNumber of [1, 5, 20, 23, 24, 31]) {
  const slide = presentation.slides.getItem(slideNumber - 1);
  console.log(`\n=== SLIDE ${slideNumber} id=${slide.id} ===`);
  slide.shapes.items.forEach((shape, shapeIndex) => {
    const row = {
      shapeIndex,
      ctor: shape.constructor?.name,
      id: shape.id,
      name: shape.name,
      position: safe(shape.position),
      text: shape.text ? String(shape.text) : undefined,
      fontSize: shape.text?.fontSize,
      typeface: shape.text?.typeface,
      color: safe(shape.text?.color),
      alignment: shape.text?.alignment,
      placeholderType: shape.placeholderType,
      geometry: shape.geometry,
      fill: safe(shape.fill),
      line: safe(shape.line),
    };
    console.log(JSON.stringify(row));
  });
  for (const chart of slide.charts.items) {
    console.log("CHART", JSON.stringify({
      id: chart.id,
      name: chart.name,
      type: chart.chartType,
      title: chart.title,
      position: safe(chart.position),
      series: chart.series.items.map((s) => ({
        name: s.name,
        categories: safe(s.categories),
        values: safe(s.values),
      })),
    }));
  }
}
