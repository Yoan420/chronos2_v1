import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePath = process.argv[2];
const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));

const apiHelp = presentation.help("*", {
  search: "slides remove delete move reorder clone duplicate",
  include: ["index", "examples", "notes"],
  maxChars: 18000,
});
console.log("===HELP===");
console.log(apiHelp.ndjson ?? apiHelp);

for (const slideNumber of [1, 5, 20, 23, 24, 31]) {
  const slide = presentation.slides.getItem(slideNumber - 1);
  const inspection = await presentation.inspect({
    target: { id: slide.id },
    kind: "slide,textbox,shape,image,table,chart,notes",
    include: "index,position,text,style,data,notes",
    maxChars: 30000,
  });
  console.log(`===SLIDE ${slideNumber}===`);
  console.log(inspection.ndjson ?? inspection);
}
