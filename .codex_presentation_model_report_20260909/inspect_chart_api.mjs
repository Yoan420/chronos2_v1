import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const presentation = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
const chart = presentation.slides.getItem(23).charts.items[0];
console.log("charts proto", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides.getItem(23).charts)));
console.log("chart proto", Object.getOwnPropertyNames(Object.getPrototypeOf(chart)));
console.log("series proto", Object.getOwnPropertyNames(Object.getPrototypeOf(chart.series)));
console.log("series item proto", Object.getOwnPropertyNames(Object.getPrototypeOf(chart.series.getItemAt(0))));
console.log("series add", String(chart.series.add));
console.log("slide delete", String(presentation.slides.getItem(0).delete));
for (const query of ["chart.series.add", "chart.delete", "slide.charts.delete", "chart.series.remove"]) {
  const result = presentation.help(query, { include: ["index", "examples", "notes"], maxChars: 12000 });
  console.log(`=== ${query} ===`);
  console.log(result.ndjson ?? result);
}
