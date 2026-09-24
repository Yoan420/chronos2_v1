import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const workspaceDir='C:/Users/BQ6757/chronos2_v1';
const root=workspaceDir+'/tmp/soutenance';
const skill='C:/Users/BQ6757/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const ref='C:/Users/BQ6757/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
process.env.RUNTIME_NODE_MODULES='C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE='C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe';
const finalPath=workspaceDir+'/output/soutenance/Soutenance_Yoan_Kesraoui_10_minutes.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const result=await finalizePresentation({workspaceDir,candidatePath:root+'/draft/candidate.pptx',finalPath,pythonExecutable:'C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','6','--require-native-table-slide','12'],requiredNativeTableOwnerSlides:[6,12],fontPolicy:{basis:'reference',families:['Helvetica Neue'],referencePath:ref,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(ref)).digest('hex')},verifyArtifactToolImport:true,receiptPath:root+'/draft/validation.json'});
console.log(JSON.stringify(result));
const p=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(root+'/final-preview',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(`${root}/final-preview/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
 console.log('Final slide '+(i+1));
}
