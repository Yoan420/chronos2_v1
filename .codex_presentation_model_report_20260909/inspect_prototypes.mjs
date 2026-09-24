import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const presentation = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
const slides = presentation.slides;
const first = slides.getItem(0);
const firstShape = first.shapes.getItemAt(0);

console.log("slides own", Object.getOwnPropertyNames(slides));
console.log("slides proto", Object.getOwnPropertyNames(Object.getPrototypeOf(slides)));
console.log("slide own", Object.getOwnPropertyNames(first));
console.log("slide proto", Object.getOwnPropertyNames(Object.getPrototypeOf(first)));
console.log("shapes own", Object.getOwnPropertyNames(first.shapes));
console.log("shapes proto", Object.getOwnPropertyNames(Object.getPrototypeOf(first.shapes)));
console.log("shape proto", Object.getOwnPropertyNames(Object.getPrototypeOf(firstShape)));

for (const query of ["presentation.slides", "slides.remove", "slides.delete", "slide.clone", "slide.duplicate"]) {
  const result = presentation.help(query, { include: ["index", "examples", "notes"], maxChars: 12000 });
  console.log(`=== ${query} ===`);
  console.log(result.ndjson ?? result);
}
