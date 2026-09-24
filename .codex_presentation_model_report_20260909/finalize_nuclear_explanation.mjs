import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const ROOT='C:/Users/BQ6757/chronos2_v1';
const BUILD=ROOT+'/.codex_presentation_nuclear_explanation_20260909';
const SKILL='C:/Users/BQ6757/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const FINAL=ROOT+'/deliverables/Presentation_Chronos2_FR_2026-09-09_integration_nucleaire.pptx';
const {finalizePresentation}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
await fs.mkdir(BUILD+'/finalizer',{recursive:true});
const result=await finalizePresentation({
 workspaceDir:ROOT,
 candidatePath:BUILD+'/candidate.pptx',
 finalPath:FINAL,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',
 explicitTotalSlideCount:6,
 requiredNativeTableOwnerSlides:[3,4,6],
 requiredNativeChartOwnerSlides:[5,6],
 requiredEmbeddedWorkbookChartOwnerSlides:[],
 nativeChartTargetApplication:'powerpoint',
 materializeLiteralChartWorkbooks:false,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','4','--require-native-table-slide','6'],
 fontPolicy:{basis:'reference',families:['Helvetica Neue'],referencePath:ROOT+'/deliverables/Presentation_Chronos2_FR_2026-09-09_avec_Kalman.pptx',referenceSha256:'b657b83921ad4389d7220127e5ab9efcd202c7156a591e73c7004fcd62fb93e8'},
 verifyArtifactToolImport:true,
 receiptPath:BUILD+'/finalizer/'+path.basename(FINAL)+'.validation.json',
});
console.log(JSON.stringify(result,null,2));
