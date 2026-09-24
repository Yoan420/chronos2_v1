import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const ROOT='C:/Users/BQ6757/chronos2_v1';
const SOURCE=ROOT+'/deliverables/Presentation_Chronos2_FR_2026-09-09_avec_Kalman.pptx';
const BUILD=ROOT+'/.codex_presentation_nuclear_explanation_20260909';
const SOURCE_HASH='b657b83921ad4389d7220127e5ab9efcd202c7156a591e73c7004fcd62fb93e8';
if(crypto.createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')!==SOURCE_HASH)throw new Error('Source changed; inspect the latest user edits first');
const pres=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
if(pres.slides.count!==5)throw new Error('Expected five user-edited slides');
const retained=pres.slides.items.slice(2).map(s=>({slide:s,shapes:JSON.stringify(s.shapes.items.map(sh=>({text:String(sh.text??''),position:sh.position})))}));
const FONT='Helvetica Neue',BLUE='#3D8DFF',KALMAN='#2468B8',BLACK='#000000',GREY='#545A64';
function text(shape,value,size=20,opts={}){
 shape.text.set(value);
 shape.text.style={typeface:FONT,fontSize:size,fill:BLACK,autoFit:'none',wrap:'square',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...opts};
 return shape;
}
function tx(slide,value,left,top,width,height,size=20,opts={}){
 return text(slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}}),value,size,opts);
}
function find(slide,value){const s=slide.shapes.items.find(s=>String(s.text??'')===value);if(!s)throw new Error('Missing source text: '+value);return s;}
function notes(slide,value){slide.speakerNotes.textFrame.setText(value);slide.speakerNotes.setVisible(true);}

const [pipeline,models]=pres.slides.items;
text(find(pipeline,'Prix récents\nCalendrier\nCharges résiduelles\nFR, DE, BE, NL, ES'),'Prix récents\nCalendrier\nCharges résiduelles\nFR, DE, BE, NL, ES\nNucléaire FR prévu',17,{fill:GREY});
notes(pipeline,String(pipeline.speakerNotes.textFrame.text??'')+'\n\nPérimètre de ce support : variante nucléaire FR du rapport du 09/09/2026. La production nucléaire prévue enrichit les entrées de Chronos-2, du correcteur CatBoost et du candidat Kalman linear_market. Le pipeline standard Both n’active pas automatiquement ce parcours expérimental. Sources : NUCLEAR_FORECAST.md, config/nuclear_forecast.yaml.');

text(find(models,'Chronos-2 fournit la base, le correcteur apprend les erreurs récurrentes et Kalman suit les biais récents.'),'Variante nucléaire FR : Chronos-2 fournit la base, CatBoost corrige ses erreurs, puis Kalman suit le biais récent.',20,{fill:GREY});
const chronosBody=find(models,'Jusqu’à 2 048 h de prix passés.\n\nCalendrier et charges résiduelles\ndes cinq pays, connus à D−1.\n\nProduit une distribution de prix.');
chronosBody.position={...chronosBody.position,height:164};
text(chronosBody,'Jusqu’à 2 048 h de prix passés.\n\nCalendrier, charges résiduelles\ndes cinq pays et nucléaire FR\nprévu, en entrées connues à D−1.\n\nProduit une distribution de prix.',18);
text(find(models,'Cible : prix réel − P50 Chronos-2\n\nCatBoost et HistGradientBoosting\najoutent des arbres successifs\npour réduire les erreurs.'),'Cible : prix réel − P50 Chronos-2\n\nCatBoost ajoute des arbres\nsuccessifs pour réduire les erreurs.\n\nRéentraînement chaque jour\nsur les 365 jours précédents.',18);
text(find(models,'CatBoost 50 %\nHistGradientBoosting 50 %'),'Variante nucléaire\nCatBoost seul',18,{bold:true,alignment:'center'});
notes(models,'Le support décrit ici la variante nucléaire FR illustrée par le rapport du 09/09/2026. La configuration résolue utilise residual_correction.backend: catboost, et non le mélange 50/50 CatBoost/HistGradientBoosting du support général.\nChronos-2 reste un réseau préentraîné gelé. Le pipeline recalcule ses prévisions historiques avec la covariable nucléaire dans le contexte et dans les entrées futures connues. Le correcteur apprend la cible prix observé moins P50 Chronos-2, sur les 365 jours civils précédant chaque livraison D. Il reçoit également la colonne nucléaire.\nLe filtre de Kalman intervient ensuite. Sa banque comprend un candidat linear_market qui reçoit le nucléaire normalisé. La gouvernance peut choisir un autre filtre ou un poids nul. Le gain de la correction globale ne mesure pas le gain de la seule variable nucléaire.\nL’exemple 150 + 8 + 3 = 161 EUR/MWh est fictif. Les corrections décalent P10, P50 et P90 ensemble à une heure donnée.\nSources : runs/experiments/nuclear_forecast_v1/2026-09-09/fr/civil_pit_v2/resolved_config.yaml, section residual_correction ; NUCLEAR_FORECAST.md, Méthode et lecture des performances ; chronos2_hourly/nuclear_forecast.py ; chronos2_hourly/kalman_residual.py.');

const nuclear=models.duplicate();
nuclear.moveTo(2);
for(const s of [...nuclear.shapes.items])nuclear.shapes.deleteById(s.id);
tx(nuclear,'Intégration de la production nucléaire FR',41,36,1197,66,39,{bold:true});
tx(nuclear,'Une prévision horaire de production en GW, issue de Saturn.\nSnapshot demandé à D−1 08 h (Paris), aligné sur les heures de livraison.',41,111,1197,65,21,{fill:GREY});
const values=[
 ['Chronos-2','Série supplémentaire dans le contexte historique et le futur connu.\nPoids du réseau gelés, prévisions historiques recalculées.'],
 ['Correcteur CatBoost','Colonne explicative pour apprendre l’erreur de Chronos-2.\nRéentraînement quotidien sur les 365 jours précédents.'],
 ['Kalman « marché »','Entrée normalisée d’un filtre candidat.\nLa gouvernance choisit le filtre et son poids de correction.'],
];
const tb=nuclear.tables.add({rows:3,columns:2,left:41,top:206,width:1197,height:255,columnWidths:[286,911],values});
tb.styleOptions={headerRow:false,bandedRows:false,firstColumn:false};
tb.borders.assign({fill:'#FFFFFF',width:3,style:'solid'});
tb.cells.block({row:0,column:0,rowCount:3,columnCount:2}).assign({fill:'#F2F2F2',textStyle:{typeface:FONT,fontSize:20,color:BLACK,alignment:'left',verticalAlignment:'middle',autoFit:'none'},margins:{left:16,right:14,top:10,bottom:10},anchor:'center'});
for(let row=0;row<3;row++)tb.getCell(row,0).text.style={typeface:FONT,fontSize:22,bold:true,fill:row===2?KALMAN:BLUE};
tx(nuclear,'Ce qui change par rapport à l’ancien essai identifié',41,491,1197,40,26,{bold:true});
tx(nuclear,'Ancien essai : nucléaire dans le correcteur seul, prévisions Chronos existantes réutilisées.\nParcours actuel : plusieurs étages utilisent la variable, avec un protocole de calcul différent.',41,540,1197,60,20,{fill:GREY});
tx(nuclear,'Le gain propre du nucléaire reste à confirmer par un test\navec / sans la variable, à données et protocole identiques.',41,629,1197,65,25,{bold:true,fill:KALMAN});
notes(nuclear,'Explication orale : pour chaque heure de la journée à prévoir, on ajoute une valeur de production nucléaire française prévue en GW. La source Saturn est power.fr.generation.nuclear.gw.fcst. L’alias est fr_nuclear_generation_fcst_gw, avec la colonne connue future known_fr_nuclear_generation_fcst_gw_oracle. Ici oracle est un nom technique : il désigne une prévision sélectionnée as-of, pas la production réalisée. Le timestamp D−1 08 h Europe/Paris est celui de la requête historique. Il ne constitue pas à lui seul une preuve de la date de publication initiale du fournisseur. Le parcours reste expérimental.\n\nChronos-2 : la série enrichit le contexte historique et les covariables futures connues, aux côtés des charges résiduelles et du calendrier. Le réseau garde ses poids préentraînés, mais les prévisions Chronos historiques sont recalculées avec cette nouvelle entrée.\n\nCatBoost : la même variable entre dans les colonnes explicatives. La cible est prix observé − P50 Chronos. On réentraîne le correcteur chaque jour sur D−365 à D−1, en utilisant les nouvelles prévisions Chronos. Le run nucléaire actuel utilise CatBoost seul, pas le mélange 50/50 du pipeline de référence.\n\nKalman : le candidat linear_market reçoit le nucléaire centré et normalisé. La normalisation utilise la médiane et max(1,4826 × MAD, écart-type, 10⁻⁶), puis borne la valeur à ±5. L’état du filtre apprend un coefficient avec les autres variables. La gouvernance choisit le candidat et son poids à partir des erreurs passées, donc ce candidat peut ne pas être retenu. Il n’existe pas de règle fixe de conversion d’un GW nucléaire en EUR/MWh. Les agrégats de charges résiduelles ne changent pas.\n\nAncien essai FR identifié : runs/tmp/screen_fundamentals_nested.py réutilisait les experts OOF existants sans relancer Chronos, puis ajoutait notamment le nucléaire dans des variantes du correcteur. Il utilisait l’ancien cache fr_nuclear_generation_fcst_long.parquet, que le parcours actuel exclut en raison d’une autre interprétation horaire. Cet essai utilisait déjà un forecast de génération. On ne peut donc pas expliquer tous les résultats précédents par une confusion avec la disponibilité REMIT. Aucun score FR antérieur chiffré n’a été confirmé dans les résultats retrouvés. Le constat d’absence de gain est rapporté par l’utilisateur.\n\nInterprétation : le nouveau parcours diffère sur plusieurs dimensions. Une amélioration globale ne permet donc pas d’attribuer le gain à la seule variable nucléaire. Il faut refaire le même pipeline avec et sans le nucléaire, sur les mêmes données, périodes, heures, hyperparamètres et règles de sélection, puis comparer les erreurs appariées hors échantillon. Le graphique d’influence du rapport mesure une sensibilité locale de la prévision amont, pas un gain de MAE ni une relation causale.\n\nSources : NUCLEAR_FORECAST.md, sections Protocole civil PIT et Méthode et lecture des performances ; run_nuclear_forecast.py, injection covariables ; chronos2_hourly/nuclear_forecast.py, _nuclear_kalman_config et refit rolling ; chronos2_hourly/models/residual_corrector.py ; chronos2_hourly/kalman_residual.py ; runs/experiments/nuclear_forecast_v1/2026-09-09/fr/civil_pit_v2/resolved_config.yaml ; runs/tmp/screen_fundamentals_nested.py.');

for(const item of retained){
 if(JSON.stringify(item.slide.shapes.items.map(sh=>({text:String(sh.text??''),position:sh.position})))!==item.shapes)throw new Error('Unrelated report slide modified');
}
if(pres.slides.count!==6)throw new Error('Expected six final slides');
await fs.mkdir(BUILD,{recursive:true});
await (await PresentationFile.exportPptx(pres)).save(BUILD+'/candidate.pptx');
console.log(JSON.stringify({candidate:BUILD+'/candidate.pptx',slides:pres.slides.count,changedSlides:[1,2],addedSlides:[3]},null,2));
