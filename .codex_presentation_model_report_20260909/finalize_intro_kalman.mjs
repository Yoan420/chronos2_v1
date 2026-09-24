import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const ROOT='C:/Users/BQ6757/chronos2_v1';
const BUILD=ROOT+'/.codex_presentation_intro_kalman_20260909';
const SKILL='C:/Users/BQ6757/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const FINAL=ROOT+'/deliverables/Presentation_Chronos2_FR_2026-09-09_avec_Kalman.pptx';
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
 requiredNativeTableOwnerSlides:[4,6],
 requiredNativeChartOwnerSlides:[5,6],
 requiredEmbeddedWorkbookChartOwnerSlides:[],
 nativeChartTargetApplication:'powerpoint',
 materializeLiteralChartWorkbooks:false,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','4','--require-native-table-slide','6'],
 fontPolicy:{basis:'reference',families:['Helvetica Neue'],referencePath:ROOT+'/deliverables/Presentation_Chronos2_FR_2026-08-26_completee_rapport_2026-09-09.pptx',referenceSha256:'5043a137849f35fa8cf2273133f1c96f013a25f920696eff65723c636b9b7975'},
 verifyArtifactToolImport:true,
 receiptPath:BUILD+'/finalizer/'+path.basename(FINAL)+'.validation.json',
});
console.log(JSON.stringify(result,null,2));
