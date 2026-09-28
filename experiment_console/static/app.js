'use strict';
const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {route:'',primary:{days:[],warnings:[]},runInfo:null,runRevision:0,boot:null,selectedDay:null,launchDay:null,refreshing:false,loaded:false,launching:false,
  regional:{day:null,country:'FR',preflight:null,checking:false,launching:false,evaluationPreflight:null,evaluationChecking:false,evaluationLaunching:false,
    annualCountry:'FR',annualPreflight:null,annualChecking:false,
    runs:[],detail:null,statusDoc:null,receiptDoc:null,logs:null,error:null,pinnedRunId:null,refreshing:false,lastCompletedEvaluationId:null}};
const warnings = (items, kind='') => items?.length ? `<div class="notice ${kind}">${items.map(esc).join('<br>')}</div>` : '';
const formatDay = value => value ? new Date(`${value}T12:00:00`).toLocaleDateString('fr-FR',{day:'numeric',month:'long',year:'numeric'}) : 'Date indisponible';
const formatTime = value => value ? new Date(value).toLocaleString('fr-FR',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}) : 'Non disponible';
const artifactUrl = (file, download=false) => `/api/primary-artifact?path=${encodeURIComponent(file.path)}${download?'&download=1':''}`;
const runArtifactUrl = (id,path,download=false) => `/api/runs/${encodeURIComponent(id)}/artifact?path=${encodeURIComponent(path)}${download?'&download=1':''}`;
const statusLabels = {queued:'En attente',starting:'Démarrage',running:'Calcul en cours',succeeded:'Calcul terminé',failed:'Calcul échoué',cancelling:'Arrêt demandé',cancelled:'Calcul arrêté',interrupted:'Calcul interrompu'};
function toast(message){const target=$('#toast');target.textContent=message;target.hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>target.hidden=true,8000);}
async function api(path, body, retry=true, extraHeaders={}){
  const options={cache:'no-store'};
  if(body!==undefined){options.method='POST';options.headers={'Content-Type':'application/json','X-Console-Token':state.boot?.token||'',...extraHeaders};options.body=JSON.stringify(body);}
  const response=await fetch(path,options);
  let value;
  try{value=await response.json();}catch{throw Error('Réponse illisible du serveur local.');}
  if(response.status===403&&body!==undefined&&retry&&value.error?.startsWith('Session expirée')){state.boot=await api('/api/bootstrap');return api(path,body,false,extraHeaders);}
  if(!response.ok){const error=Error(value.error||`Erreur ${response.status}`);error.httpStatus=response.status;throw error;}
  return value;
}
function connection(ok){const target=$('#connection');target.className=`connection ${ok?'':'offline'}`;target.textContent=ok?'À jour · suivi automatique':'Connexion perdue · nouvelle tentative automatique';}
function empty(title,text){return `<div class="empty"><span class="empty-icon" aria-hidden="true">◇</span><strong>${esc(title)}</strong><p>${esc(text)}</p></div>`;}
function pageHead(eyebrow,title,subtitle,actions=''){return `<div class="page-head"><div><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1><p class="subtitle">${esc(subtitle)}</p></div><div class="actions">${actions}</div></div>`;}
function selectedDelivery(){return state.primary.days.find(day=>day.date===state.selectedDay);}
function datePicker(){return state.primary.days.length?`<label class="delivery-picker">Date de livraison<select id="delivery-day">${state.primary.days.map(day=>`<option value="${esc(day.date)}" ${day.date===state.selectedDay?'selected':''}>${esc(formatDay(day.date))}</option>`).join('')}</select></label>`:'';}
function reportActions(file,title){return `<button data-report="${esc(file.path)}" data-title="${esc(title)}">Agrandir ↗</button><a class="primary-download" href="${esc(artifactUrl(file,true))}" download="${esc(file.name)}">Télécharger HTML ↓</a>`;}
function zoneCard(zone,day){
  const labels={FR:'France',DE:'Allemagne',BE:'Belgique',NL:'Pays-Bas'};
  const publication=day.zones.find(item=>item.zone===zone);
  const report=publication?.report, csv=publication?.csv;
  const status=publication?.publication_status;
  const label=!publication?'Non publié':status==='verified'?'Publication vérifiée':status==='incomplete'?'Publication incomplète':'Intégrité non vérifiée';
  const updated=report?.updated_at||csv?.updated_at;
  return `<article class="primary-zone ${!publication?'unavailable':''}"><header><div><span class="zone-code">${esc(zone)}</span><h3>${esc(publication?.label||labels[zone]||zone)}</h3></div><span class="publication-badge ${esc(status||'missing')}" title="La vérification porte sur l’intégrité des fichiers publiés.">${esc(label)}</span></header>
    <p>${report?'Prévisions NYX et analyse détaillée.':csv?'Prévisions disponibles au format CSV.':'Aucun résultat publié pour cette livraison.'}</p>
    <div class="primary-zone-actions">${report?`<button data-report="${esc(report.path)}" data-title="NYX · ${esc(publication.label)} · ${esc(formatDay(day.date))}">Ouvrir le rapport ↗</button>`:''}${csv?`<a class="primary-download" href="${esc(artifactUrl(csv,true))}" download="${esc(csv.name)}">CSV ↓</a>`:''}${report?`<a class="primary-download" href="${esc(artifactUrl(report,true))}" download="${esc(report.name)}">HTML ↓</a>`:''}</div>
    ${updated?`<span class="publication-time">Mis à jour le ${esc(formatTime(updated))}</span>`:''}${publication?.warnings?.length?`<details class="publication-details"><summary>Détails de la publication</summary>${warnings(publication.warnings)}</details>`:''}</article>`;
}
function renderDashboard(){
  const day=selectedDelivery();
  const main=$('#main');
  if(!main.nyxDashboard){
    main.innerHTML=pageHead('NYX / PRÉVISIONS','Résultats principaux','Le rapport CWE et les prévisions NYX par pays.',`<div id="dashboard-date-picker">${datePicker()}</div>` )+'<section id="primary-run-panel" aria-label="Lancer et suivre la prévision"></section><div id="primary-results-content"></div>';
    main.nyxDashboard=true;
  }else $('#dashboard-date-picker').innerHTML=datePicker();
  renderRunPanel();
  const target=$('#primary-results-content');
  if(!day){target.innerHTML=empty('Aucun résultat publié','Lancez une prévision. Les rapports apparaîtront ici dès leur publication.')+warnings(state.primary.warnings);return;}
  const countries=[...new Set(['FR','DE','BE','NL',...day.zones.map(zone=>zone.zone)])];
  target.innerHTML=`${warnings(state.primary.warnings)}<section class="primary-cwe"><div class="section-heading"><div><div class="eyebrow">${esc(formatDay(day.date))}</div><h2>Rapport global CWE</h2></div><div class="actions">${day.cwe?reportActions(day.cwe,'CWE · '+formatDay(day.date)):''}</div></div>
    ${day.cwe?`<iframe class="primary-report-frame" data-cwe-report data-delivery-day="${esc(day.date)}" title="Rapport global CWE du ${esc(formatDay(day.date))}" sandbox="allow-scripts" src="${esc(artifactUrl(day.cwe))}"></iframe><p class="publication-time">Mis à jour le ${esc(formatTime(day.cwe.updated_at))} · Le rapport rassemble les pays disponibles.</p>`:empty('Rapport CWE non publié','Le rapport global n’est pas disponible pour cette date. Consultez les résultats par pays ci-dessous.')}</section>
    <div class="section-heading"><h2>Résultats par pays</h2><span class="secondary">NYX · ${esc(formatDay(day.date))}</span></div><section class="primary-zone-grid" aria-label="Prévisions par pays">${countries.map(zone=>zoneCard(zone,day)).join('')}</section>
    <p class="primary-sources">Publications locales : <code>runs/reports</code> et <code>runs/exports</code></p>`;
}
function renderPublications(){
  const days=state.primary.days;
  $('#main').innerHTML=pageHead('NYX / PUBLICATIONS','Publications','Retrouvez les résultats principaux par date de livraison.')+warnings(state.primary.warnings)+(days.length?`<div class="panel table-wrap"><table><thead><tr><th>Livraison</th><th>Rapport CWE</th><th>Pays disponibles</th><th></th></tr></thead><tbody>${days.map(day=>`<tr><td><strong>${esc(formatDay(day.date))}</strong></td><td>${day.cwe?'Disponible':'Non publié'}</td><td><div class="publication-countries">${day.zones.map(zone=>`<span>${esc(zone.zone)}</span>`).join('')||'Aucun'}</div></td><td><button data-delivery="${esc(day.date)}">Voir les résultats →</button></td></tr>`).join('')}</tbody></table></div>`:empty('Aucune publication disponible','Les prochains résultats NYX apparaîtront ici.'));
}
function renderRunPanel(){
  const panel=$('#primary-run-panel');if(!panel)return;
  const info=state.runInfo;
  const signature=JSON.stringify([info,state.launching]);
  if(panel.nyxSignature===signature)return;
  panel.nyxSignature=signature;
  if(!info){
    if(panel.nyxReady){$('#launch-forecast').disabled=true;$('#run-launch-status').innerHTML=warnings(['Suivi temporairement indisponible. Reconnexion automatique.']);}
    else panel.innerHTML='<div class="run-launch-panel"><p class="muted">Connexion au calcul NYX…</p></div>';
    return;
  }
  const run=info.active||info.latest;
  const busy=Boolean(info.active)||state.launching;
  if(state.launchDay===null)state.launchDay=info.defaults?.delivery_day||'';
  // Keep the date control mounted while progress polls update the status.
  if(!panel.nyxReady){
    panel.innerHTML=`<div class="run-launch-panel"><div class="run-launch-intro"><div><div class="eyebrow">NOUVELLE PRÉVISION</div><h2>Calculer avec NYX</h2><p>Actualise les sources, calcule les prévisions BE, DE, FR et NL, puis publie les rapports.</p></div><div class="run-launch-actions"><label class="launch-date-picker" for="launch-delivery-day">Date de livraison à prévoir<input type="date" id="launch-delivery-day" required value="${esc(state.launchDay)}"></label><button id="launch-forecast" class="primary launch-forecast" data-launch>▷ Lancer la prévision</button></div></div><div id="run-launch-status"></div></div>`;
    panel.nyxReady=true;
  }
  $('#launch-delivery-day').disabled=busy||info.available===false;
  const button=$('#launch-forecast');
  button.disabled=busy||info.available===false;
  button.textContent=state.launching?'Démarrage…':busy?'◌ Prévision en cours':'▷ Lancer la prévision';
  const progress=run?.progress;
  $('#run-launch-status').innerHTML=`${info.warning?warnings([info.warning]):''}${run?`<div class="primary-run-status" role="status"><span class="badge ${esc(run.status)}">${esc(statusLabels[run.status]||run.status)}</span><span>${run.delivery_day?'Livraison du '+esc(formatDay(run.delivery_day)):''}</span><span class="publication-time">${run.finished_at?'Terminé le '+esc(formatTime(run.finished_at)):run.started_at?'Démarré le '+esc(formatTime(run.started_at)):run.created_at?'Demandé le '+esc(formatTime(run.created_at)):''}</span>${progress?.phase?`<span class="run-phase">${esc(progress.phase)}</span>`:''}</div>${renderProgress(run)}${run.error?warnings([run.error],'error'):''}`:'<p class="publication-time">Le calcul continue si vous fermez la fenêtre.</p>'}`;
}
function renderProgress(run){
  const progress=run.progress||{};
  const known=Number.isFinite(progress.percent)&&Number.isFinite(progress.completed)&&Number.isFinite(progress.total)&&progress.total>0;
  const active=['starting','running','cancelling'].includes(run.status);
  if(!known&&!active)return '';
  const percent=known?Math.max(0,Math.min(100,progress.percent)):null;
  const caption=known?`${progress.completed} / ${progress.total} étapes traitées`:'Préparation du suivi des étapes…';
  const stepLabels={pending:'En attente',queued:'En attente',running:'En cours',complete:'Terminé',failed:'Échec',skipped:'Non exécuté',interrupted:'Interrompu'};
  return `<div class="run-progress ${esc(run.status)}"><div class="run-progress-caption"><span>${esc(caption)}</span>${known?`<strong>${esc(percent)} %</strong>`:''}</div><div class="run-progress-track ${known?'':'indeterminate'}" role="progressbar" aria-label="Avancement de la prévision NYX" aria-valuemin="0" aria-valuemax="100" ${known?`aria-valuenow="${percent}" aria-valuetext="${esc(caption)}"`:''}><span class="run-progress-fill" ${known?`style="width:${percent}%"`:''}></span></div>${progress.steps?.length?`<ol class="run-progress-steps">${progress.steps.map(step=>`<li class="${esc(step.status)}" ${step.status==='running'?'aria-current="step"':''}><span class="step-indicator" aria-hidden="true">${step.status==='complete'?'✓':step.status==='failed'?'!':step.status==='running'?'◌':'·'}</span><span><strong>${esc(step.zone||step.label||step.name)}</strong><small>${esc(stepLabels[step.status]||step.status)}</small></span></li>`).join('')}</ol>`:''}</div>`;
}
async function launchForecast(){
  if(!state.runInfo||state.launching||state.runInfo.active||state.runInfo.available===false)return;
  const dateInput=$('#launch-delivery-day');
  if(dateInput&&!dateInput.reportValidity())return;
  const deliveryDay=state.launchDay;
  const parsed=new Date(`${deliveryDay}T00:00:00Z`);
  if(!/^\d{4}-\d{2}-\d{2}$/.test(deliveryDay)||deliveryDay.startsWith('0000')||!Number.isFinite(parsed.valueOf())||parsed.toISOString().slice(0,10)!==deliveryDay){toast('Choisissez une date de livraison valide.');return;}
  state.launching=true;renderRunPanel();
  try{
    if(!state.boot)state.boot=await api('/api/bootstrap');
    let previous;
    try{previous=JSON.parse(sessionStorage.getItem('nyx-launch-intent'));}catch{/* Old clients stored only a key. */}
    const intent=previous?.delivery_day===deliveryDay&&typeof previous.key==='string'&&previous.key?previous.key:crypto.randomUUID();
    sessionStorage.setItem('nyx-launch-intent',JSON.stringify({key:intent,delivery_day:deliveryDay}));
    const accepted=await api('/api/primary-run',{delivery_day:deliveryDay},true,{'Idempotency-Key':intent});
    const active=['queued','starting','running','cancelling'].includes(accepted.run.status);
    state.runRevision++;
    state.runInfo={...state.runInfo,latest:accepted.run,active:active?accepted.run:null};
    sessionStorage.removeItem('nyx-launch-intent');
    toast('La prévision NYX a été demandée. Son avancement apparaît ici.');
  }catch(error){toast(error.message);}
  finally{state.launching=false;renderRunPanel();}
}
function showReport(path,title){
  const dialog=$('#dialog');
  const cweDay=/^reports\/model_storm\/CWE_Model_Storm_(\d{4}-\d{2}-\d{2})\.html$/.exec(path)?.[1];
  dialog.className='report-dialog';
  dialog.innerHTML=`<div class="section-heading"><h2 id="dialog-title">${esc(title||'Résultat NYX')}</h2><button data-close aria-label="Fermer le rapport">Fermer ✕</button></div><iframe class="report-frame" ${cweDay?`data-cwe-report data-delivery-day="${esc(cweDay)}"`:''} title="${esc(title||'Résultat NYX')}" sandbox="allow-scripts" src="${esc(artifactUrl({path}))}"></iframe>`;
  if(!dialog.open)dialog.showModal();
}
function handleReportNavigation(event){
  const message=event.data;
  if(!message||typeof message!=='object'||!['nyx:delivery-ready','nyx:delivery-selected'].includes(message.type))return;
  const frame=$$('iframe[data-cwe-report]').find(item=>item.contentWindow===event.source);
  if(!frame)return;
  if(message.type==='nyx:delivery-ready'){
    const focusTarget=$('#dialog').contains(frame)?'dialog':'main';
    const focus=state.reportFocus===focusTarget;
    if(focus)state.reportFocus=null;
    frame.contentWindow.postMessage({type:'nyx:delivery-options',dates:state.primary.days.map(day=>day.date),selectedDay:frame.dataset.deliveryDay,focus},'*');
    return;
  }
  if(typeof message.date!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(message.date))return;
  const day=state.primary.days.find(item=>item.date===message.date);
  if(!day)return;
  const dialog=$('#dialog');
  const inDialog=dialog.contains(frame);
  state.selectedDay=day.date;
  state.reportFocus=inDialog&&day.cwe?'dialog':'main';
  if(state.route==='dashboard')renderDashboard();
  else location.hash='dashboard';
  if(inDialog){
    if(day.cwe)showReport(day.cwe.path,'CWE · '+formatDay(day.date));
    else dialog.close();
  }
  if(!day.cwe){state.reportFocus=null;$('#delivery-day')?.focus();}
}
function regionalSelection(){return `${state.regional.day}|${state.regional.country}`;}
function regionalActive(){return state.regional.runs.some(run=>['queued','starting','running','cancelling'].includes(run.status));}
function renderRegional(){
  const main=$('#main');
  const regional=state.regional;
  if(!main.nyxRegional){
    const annualPanel=`<section class="panel padded regional-panel" aria-label="Modèles annuels CWE historiques"><div class="section-heading"><div><div class="eyebrow">ÉTUDE ANNUELLE · 2026-09-23</div><h2>Modèles annuels FR / BE / NL</h2></div></div><p>Consultez le prix retenu et la probabilité de prix négatif sur l’année étudiée. Cette comparaison est rétrospective : aucune prévision future n’est activée ici.</p><div class="regional-controls"><label>Pays historique<select id="annual-cwe-country">${['FR','BE','NL'].map(zone=>`<option value="${zone}" ${zone===regional.annualCountry?'selected':''}>${zone}</option>`).join('')}</select></label><button id="annual-cwe-check" data-annual-cwe-check>Vérifier les archives</button></div><div id="annual-cwe-status" role="status"></div></section>`;
    main.innerHTML=pageHead('NYX / MODÈLES RÉGIONAUX','Prévisions par pays et archives annuelles','Prix horaire et probabilité de prix négatif.')+annualPanel+
      `<section class="panel padded regional-panel" aria-label="Évaluation de la recette CPU"><div class="section-heading"><div><div class="eyebrow">VALIDATION</div><h2>Évaluer la recette CPU</h2></div></div><p>Synchronise Saturn, rejoue les origines historiques et produit un reçu. Si les critères passent, les prévisions par pays deviennent disponibles.</p><button id="regional-evaluate" data-regional-evaluate disabled>▷ Évaluer la recette CPU</button><div id="regional-evaluation-check" role="status"></div></section>`+
      `<section class="panel padded regional-panel" aria-label="Lancement régional"><div class="section-heading"><div><div class="eyebrow">NOUVELLE RECETTE CPU</div><h2>Calculer un pays</h2></div></div><p>Le moteur actualise Saturn et réentraîne le modèle avant chaque prévision. Le lancement sera disponible après validation du backtest.</p><div class="regional-controls"><label>Date de livraison<input id="regional-day" type="date" required value="${esc(regional.day)}"></label><label>Pays<select id="regional-country">${['FR','DE','BE','NL'].map(zone=>`<option value="${zone}" ${zone===regional.country?'selected':''}>${zone}</option>`).join('')}</select></label><button class="primary" id="regional-launch" data-regional-launch disabled>▷ Lancer ce pays</button></div><div id="regional-check" role="status"></div></section><section class="regional-history"><div class="section-heading"><h2>Derniers lancements</h2></div><div id="regional-runs"></div></section><section id="regional-output" aria-label="Sorties du lancement régional"></section>`;
    main.nyxRegional=true;
  }
  renderRegionalStatus();
}
function renderAnnualCweStatus(){
  if(state.route!=='regional')return;
  const regional=state.regional;
  const button=$('#annual-cwe-check');
  const target=$('#annual-cwe-status');
  if(!button||!target)return;
  button.disabled=regional.annualChecking;
  button.textContent=regional.annualChecking?'Vérification…':'Vérifier les archives';
  if(regional.annualChecking){target.innerHTML='<p class="publication-time">Contrôle local des reçus, matrices et checkpoints historiques…</p>';return;}
  const inspection=regional.annualPreflight;
  if(!inspection){target.innerHTML='<p class="publication-time">Précontrôle annuel non chargé.</p>';return;}
  if(!inspection.manifest_valid){target.innerHTML=warnings((inspection.blockers||[]).map(item=>item.message||item.code||String(item)),'error');return;}
  const country=regional.annualCountry;
  const row=inspection.countries?.[country];
  if(!row){target.innerHTML=warnings(['Aucun résultat historique disponible pour ce pays.'],'error');return;}
  const metric=(value,digits=3)=>typeof value==='number'&&Number.isFinite(value)?new Intl.NumberFormat('fr-FR',{maximumFractionDigits:digits}).format(value):'—';
  const price=row.price?.score||{},negative=row.negative?.score||{};
  const verifiedEvidence=Object.values(inspection.evidence||{}).length>0&&Object.values(inspection.evidence||{}).every(item=>item.status==='verified');
  const featureFiles=Object.values(inspection.source_feature_matrices||{});
  const checkpointFiles=Object.values(inspection.checkpoints||{});
  const featureCount=featureFiles.filter(item=>item.status==='verified').length;
  const checkpointCount=checkpointFiles.filter(item=>item.verified===true).length;
  const archiveStatus=row.archived_files_verified===true?'Archives et références vérifiées sur ce poste.':`Archives incomplètes sur ce poste : ${featureCount}/${featureFiles.length} matrices et ${checkpointCount}/${checkpointFiles.length} checkpoints vérifiés.`;
  const provenance=verifiedEvidence?'Scores du rapport annuel vérifié localement.':'Scores du manifeste historique ; les reçus originaux ne sont pas tous présents sur ce poste.';
  target.innerHTML=`<div class="notice ${row.archived_files_verified?'info':'error'}">${esc(archiveStatus)}</div><p class="publication-time">${esc(provenance)} Période : 24 septembre 2025 au 23 septembre 2026. Même année que la sélection, sans validation indépendante.</p><div class="table-wrap"><table><thead><tr><th>Pays</th><th>Prix retenu</th><th>RMSE NYX</th><th>RMSE Storm</th><th>Heures gagnées</th><th>Brier négatif</th><th>AP négatif</th></tr></thead><tbody><tr><td><strong>${esc(country)}</strong></td><td>${esc(row.price?.candidate||'—')}</td><td>${esc(metric(price.rmse))}</td><td>${esc(metric(price.storm_rmse))}</td><td>${esc(typeof price.strict_win_rate==='number'?metric(price.strict_win_rate*100,2)+' %':'—')}</td><td>${esc(metric(negative.brier,5))}</td><td>${esc(metric(negative.average_precision,4))}</td></tr></tbody></table></div><div class="notice error">Prévision annuelle indisponible : les matrices et checkpoints historiques ne fournissent pas les variables futures, le réentraînement CPU équivalent des modèles de prix ni une validation indépendante. Ce panneau est en lecture seule.</div>`;
}
function renderRegionalStatus(){
  if(state.route!=='regional')return;
  renderAnnualCweStatus();
  const regional=state.regional;
  const launch=$('#regional-launch');
  if(!launch)return;
  const ready=regional.preflight?.ready===true&&regional.preflight?.recipe_status==='validated';
  const busy=regional.checking||regional.launching||regional.evaluationLaunching||regionalActive();
  launch.disabled=!ready||busy;
  launch.textContent=regional.launching?'Démarrage…':regionalActive()?'◌ Calcul régional en cours':'▷ Lancer ce pays';
  const blockers=regional.preflight?.blockers||[];
  $('#regional-check').innerHTML=regional.checking?'<p class="publication-time">Vérification de la recette, des données et du poste…</p>':
    ready?'<div class="notice info">Précontrôle réussi : recette validée et ressources locales disponibles. La synchronisation Saturn sera lancée pendant le calcul.</div>':
    warnings(blockers.length?blockers:['Backtest régional non encore validé : lancement indisponible.'],'error');
  const evaluate=$('#regional-evaluate');
  if(evaluate){
    evaluate.disabled=regional.evaluationChecking||regional.evaluationLaunching||regionalActive()||regional.evaluationPreflight?.ready!==true;
    evaluate.textContent=regional.evaluationLaunching?'Démarrage…':regionalActive()?'◌ Calcul en cours':'▷ Évaluer la recette CPU';
    $('#regional-evaluation-check').innerHTML=regional.evaluationChecking?'<p class="publication-time">Vérification du poste pour le backtest…</p>':
      regional.evaluationPreflight?.ready===true?'<p class="publication-time">Le poste peut lancer le backtest. La disponibilité de Saturn sera vérifiée pendant le calcul.</p>':
      warnings(regional.evaluationPreflight?.blockers||['Évaluation non disponible.'],'error');
  }
  const runs=regional.runs.slice(0,12);
  $('#regional-runs').innerHTML=runs.length?`<div class="panel table-wrap"><table><thead><tr><th>Type</th><th>Livraison</th><th>Pays</th><th>État</th><th>Demandé</th><th></th></tr></thead><tbody>${runs.map(run=>`<tr><td>${run.adapter_id==='nyx_regional_cpu_backtest'?'Évaluation':'Prévision'}</td><td>${esc(run.delivery_day?formatDay(run.delivery_day):'—')}</td><td>${esc(run.countries?.join(', ')||'—')}</td><td><span class="badge ${esc(run.status)}">${esc(statusLabels[run.status]||run.status)}</span></td><td>${esc(formatTime(run.created_at))}</td><td><button data-regional-run="${esc(run.id)}">Voir les sorties →</button></td></tr>`).join('')}</tbody></table></div>`:empty('Aucun lancement régional','Les calculs de cette nouvelle recette apparaîtront ici.');
  const detail=regional.detail;
  if(!detail){$('#regional-output').innerHTML=regional.error?warnings([regional.error],'error'):'';return;}
  const rawPhase=regional.statusDoc?.phase||regional.statusDoc?.stage||'';
  const phaseLabels={preflight:'Précontrôle',sync_saturn:'Synchronisation Saturn',saturn_sync:'Synchronisation Saturn',
    build_features:'Préparation des variables',fit_models:'Réentraînement CPU',publish:'Publication des prévisions',
    official_benchmark:'Référence officielle EPEX / Storm',weekly_fit:'Rejeu des origines',
    selection_weekly_fit:'Sélection hebdomadaire',confirmation_daily_fit:'Confirmation quotidienne',
    selection_sealed:'Sélection des modèles',complete:'Calcul terminé',failed:'Calcul échoué'};
  const originCount=Number.isInteger(regional.statusDoc?.completed_origins)?` · ${regional.statusDoc.completed_origins} / ${regional.statusDoc.total_origins||173} origines`:'';
  const phase=rawPhase?`${phaseLabels[rawPhase]||rawPhase}${originCount}`:detail.activity||'';
  const evaluation=detail.adapter_id==='nyx_regional_cpu_backtest';
  const files=(detail.artifacts||[]).filter(file=>evaluation?file.path.endsWith('backtest_receipt.json')||file.path.endsWith('.html')||file.path.endsWith('.csv'):
    /^zones\/(FR|DE|BE|NL)\/forecast_[a-z]{2}_\d{4}-\d{2}-\d{2}_nyx_regional_cpu\.(csv|html)$/.test(file.path));
  const outputs=(evaluation||detail.status==='succeeded')&&files.length?`<div class="regional-files">${files.map(file=>file.path.endsWith('.html')?`<button data-regional-report="${esc(file.path)}" data-run-id="${esc(detail.id)}">Ouvrir le rapport ↗</button><a href="${esc(runArtifactUrl(detail.id,file.path,true))}" download="${esc(file.name)}">HTML ↓</a>`:`<a href="${esc(runArtifactUrl(detail.id,file.path,true))}" download="${esc(file.name)}">${esc(file.name)} ↓</a>`).join('')}</div>`:detail.status==='succeeded'?'<p class="notice error">Le calcul est terminé, mais aucun résultat consultable n’a été trouvé.</p>':'';
  const qualification=Array.isArray(regional.statusDoc?.qualification_blockers)?regional.statusDoc.qualification_blockers:[];
  const qualified=evaluation&&regional.statusDoc?.qualified===true?'<div class="notice info">Backtest qualifié. Le précontrôle des prévisions sera actualisé.</div>':
    evaluation&&regional.statusDoc?.qualified===false?'<div class="notice error">Les critères du backtest ne sont pas atteints. Les prévisions restent verrouillées.</div>':'';
  const receipt=regional.receiptDoc;
  const metric=(value,digits=2)=>typeof value==='number'&&Number.isFinite(value)?new Intl.NumberFormat('fr-FR',{maximumFractionDigits:digits}).format(value):'—';
  const countryScores=evaluation&&receipt?.countries&&typeof receipt.countries==='object'?['FR','DE','BE','NL'].map(zone=>{
    const row=receipt.countries[zone];
    if(!row||typeof row!=='object')return '';
    return `<tr><td><strong>${zone}</strong></td><td>${esc(row.selected||'—')}</td><td>${esc(metric(row.confirmation?.rmse))}</td><td>${esc(metric(row.confirmation?.storm_rmse))}</td><td>${esc(typeof row.confirmation?.strict_win_rate==='number'?metric(row.confirmation.strict_win_rate*100,1)+' %':'—')}</td><td>${esc(metric(row.negative_confirmation?.brier,4))}</td></tr>`;
  }).join(''):'';
  const scores=countryScores?`<div class="regional-scores"><h3>Scores du backtest · confirmation</h3><p class="publication-time">Prix : RMSE en €/MWh face à Storm. Prix négatif : score de Brier (plus bas est meilleur). ${receipt.confirmation_first_day&&receipt.stop_day_exclusive?`Période ${esc(receipt.confirmation_first_day)} au ${esc(receipt.stop_day_exclusive)} exclu.`:''}</p><div class="table-wrap"><table><thead><tr><th>Pays</th><th>Prix retenu</th><th>RMSE NYX</th><th>RMSE Storm</th><th>Heures gagnées</th><th>Brier négatif</th></tr></thead><tbody>${countryScores}</tbody></table></div></div>`:'';
  $('#regional-output').innerHTML=`<div class="section-heading"><h2>Suivi ${evaluation?'de l’évaluation':'de la prévision'}</h2><span class="secondary">${esc(detail.countries?.join(', ')||'')} ${esc(detail.delivery_day?formatDay(detail.delivery_day):'')}</span></div><div class="panel padded"><div class="primary-run-status"><span class="badge ${esc(detail.status)}">${esc(statusLabels[detail.status]||detail.status)}</span>${phase?`<span class="run-phase">${esc(phase)}</span>`:''}</div>${qualified}${qualification.length?warnings(qualification,'error'):''}${detail.status==='failed'||detail.status==='interrupted'?warnings([regional.statusDoc?.error||detail.activity||'Le calcul a échoué. Consultez le journal ci-dessous.'],'error'):''}${scores}${outputs}<p class="publication-time">Dossier : <code>${esc(detail.output_dir||'')}</code></p>${regional.logs?.available?`<details class="regional-log" ${detail.status==='failed'?'open':''}><summary>Journal du calcul</summary><pre>${esc(regional.logs.text?.slice(-6000)||'')}</pre></details>`:''}</div>`;
}
async function checkRegional(){
  const regional=state.regional;
  const key=regionalSelection();
  regional.checking=true;regional.preflight=null;renderRegionalStatus();
  try{
    const result=await api(`/api/regional-preflight?delivery_day=${encodeURIComponent(regional.day)}&country=${encodeURIComponent(regional.country)}`);
    if(key===regionalSelection())regional.preflight=result;
  }catch(error){if(key===regionalSelection())regional.preflight={ready:false,recipe_status:'unavailable',blockers:[error.message]};}
  finally{if(key===regionalSelection()){regional.checking=false;renderRegionalStatus();}}
}
async function checkRegionalEvaluation(){
  const regional=state.regional;
  regional.evaluationChecking=true;regional.evaluationPreflight=null;renderRegionalStatus();
  try{regional.evaluationPreflight=await api('/api/regional-evaluation-preflight');}
  catch(error){regional.evaluationPreflight={ready:false,blockers:[error.message]};}
  finally{regional.evaluationChecking=false;renderRegionalStatus();}
}
async function checkAnnualCwe(){
  const regional=state.regional;
  const country=regional.annualCountry;
  regional.annualChecking=true;regional.annualPreflight=null;renderAnnualCweStatus();
  try{
    const result=await api(`/api/annual-cwe-preflight?country=${encodeURIComponent(country)}`);
    if(country===regional.annualCountry)regional.annualPreflight=result;
  }catch(error){
    if(country===regional.annualCountry)regional.annualPreflight={manifest_valid:false,blockers:[{message:error.message}]};
  }finally{
    if(country===regional.annualCountry){regional.annualChecking=false;renderAnnualCweStatus();}
  }
}
async function refreshRegional(){
  const regional=state.regional;
  if(state.route!=='regional'||regional.refreshing)return;
  regional.refreshing=true;
  try{
    const listing=await api('/api/runs');
    regional.runs=(listing.runs||[]).filter(run=>['nyx_regional_cpu','nyx_regional_cpu_backtest'].includes(run.adapter_id));
    const completedEvaluation=regional.runs.find(run=>run.adapter_id==='nyx_regional_cpu_backtest'&&['succeeded','failed','interrupted'].includes(run.status));
    if(completedEvaluation&&completedEvaluation.id!==regional.lastCompletedEvaluationId){
      regional.lastCompletedEvaluationId=completedEvaluation.id;
      void checkRegional();
    }
    const selected=regional.runs.find(run=>run.id===regional.pinnedRunId)||regional.runs[0];
    if(selected){
      const detail=await api(`/api/runs/${encodeURIComponent(selected.id)}`);
      regional.detail=detail;regional.statusDoc=null;regional.receiptDoc=null;regional.logs=null;
      if(detail.artifacts?.some(file=>file.path==='status.json')){
        try{regional.statusDoc=await api(runArtifactUrl(detail.id,'status.json'));}catch{/* The worker may still be writing status.json. */}
      }
      if(detail.adapter_id==='nyx_regional_cpu_backtest'&&detail.artifacts?.some(file=>file.path==='results/backtest_receipt.json')){
        try{regional.receiptDoc=await api(runArtifactUrl(detail.id,'results/backtest_receipt.json'));}catch{/* The backtest may still be writing its receipt. */}
      }
      if(['starting','running','failed','interrupted'].includes(detail.status)){
        try{regional.logs=await api(`/api/runs/${encodeURIComponent(detail.id)}/logs`);}catch{/* Status remains visible. */}
      }
    }else regional.detail=null;
    regional.error=null;
  }catch(error){regional.error=error.message;}
  finally{regional.refreshing=false;renderRegionalStatus();}
}
async function launchRegional(){
  const regional=state.regional;
  if(regional.launching||regional.checking||regionalActive())return;
  if(!$('#regional-day').reportValidity())return;
  const deliveryDay=regional.day,country=regional.country;
  const parsed=new Date(`${deliveryDay}T00:00:00Z`);
  if(!/^\d{4}-\d{2}-\d{2}$/.test(deliveryDay)||deliveryDay.startsWith('0000')||!Number.isFinite(parsed.valueOf())||parsed.toISOString().slice(0,10)!==deliveryDay){toast('Choisissez une date de livraison valide.');return;}
  regional.launching=true;renderRegionalStatus();
  try{
    await checkRegional();
    if(regionalSelection()!==`${deliveryDay}|${country}`||regional.preflight?.ready!==true||regional.preflight?.recipe_status!=='validated')throw Error(regional.preflight?.blockers?.join(' ')||'Le précontrôle régional bloque le lancement.');
    if(!state.boot)state.boot=await api('/api/bootstrap');
    let intent;try{intent=JSON.parse(sessionStorage.getItem('nyx-regional-intent'));}catch{/* Ignore old session data. */}
    if(intent?.delivery_day!==deliveryDay||intent?.country!==country||!intent?.key)intent={key:crypto.randomUUID(),delivery_day:deliveryDay,country};
    if(!intent.plan_id){
      const plan=await api('/api/preview',{adapter_id:'nyx_regional_cpu',config_id:'regional_cpu_country',model:'regional_price_and_negative_probability',parameters:{delivery_day:deliveryDay,country},name:`NYX régional CPU · ${country} · ${deliveryDay}`});
      intent.plan_id=plan.id;
      sessionStorage.setItem('nyx-regional-intent',JSON.stringify(intent));
    }
    const accepted=await api('/api/launch',{plan_id:intent.plan_id,idempotency_key:intent.key});
    sessionStorage.removeItem('nyx-regional-intent');
    regional.runs=[accepted,...regional.runs.filter(run=>run.id!==accepted.id)];
    regional.pinnedRunId=accepted.id;regional.detail=accepted;
    toast('Le calcul régional a été demandé. Son avancement apparaît ici.');
    await refreshRegional();
  }catch(error){
    if([400,404].includes(error.httpStatus))sessionStorage.removeItem('nyx-regional-intent');
    toast(error.message);regional.error=error.message;
  }
  finally{regional.launching=false;renderRegionalStatus();}
}
async function launchRegionalEvaluation(){
  const regional=state.regional;
  if(regional.evaluationLaunching||regional.evaluationChecking||regionalActive())return;
  regional.evaluationLaunching=true;renderRegionalStatus();
  try{
    await checkRegionalEvaluation();
    if(regional.evaluationPreflight?.ready!==true)throw Error(regional.evaluationPreflight?.blockers?.join(' ')||'Le précontrôle de l’évaluation bloque le lancement.');
    if(!state.boot)state.boot=await api('/api/bootstrap');
    let intent;try{intent=JSON.parse(sessionStorage.getItem('nyx-regional-evaluation-intent'));}catch{/* Ignore old session data. */}
    if(!intent?.key)intent={key:crypto.randomUUID()};
    if(!intent.plan_id){
      const plan=await api('/api/preview',{adapter_id:'nyx_regional_cpu_backtest',config_id:'regional_cpu_evaluation',model:'regional_cpu_backtest',name:'NYX régional CPU · évaluation'});
      intent.plan_id=plan.id;
      sessionStorage.setItem('nyx-regional-evaluation-intent',JSON.stringify(intent));
    }
    const accepted=await api('/api/launch',{plan_id:intent.plan_id,idempotency_key:intent.key});
    sessionStorage.removeItem('nyx-regional-evaluation-intent');
    regional.runs=[accepted,...regional.runs.filter(run=>run.id!==accepted.id)];
    regional.pinnedRunId=accepted.id;regional.detail=accepted;
    toast('Le backtest régional a été demandé. Son avancement apparaît ici.');
    await refreshRegional();
  }catch(error){
    if([400,404].includes(error.httpStatus))sessionStorage.removeItem('nyx-regional-evaluation-intent');
    toast(error.message);regional.error=error.message;
  }
  finally{regional.evaluationLaunching=false;renderRegionalStatus();}
}
function showRegionalReport(runId,path){
  const dialog=$('#dialog');
  dialog.className='report-dialog';
  dialog.innerHTML=`<div class="section-heading"><h2 id="dialog-title">Rapport régional NYX</h2><button data-close aria-label="Fermer le rapport">Fermer ✕</button></div><iframe class="report-frame" title="Rapport régional NYX" sandbox="allow-scripts" src="${esc(runArtifactUrl(runId,path))}"></iframe>`;
  if(!dialog.open)dialog.showModal();
}
async function route(){
  const raw=location.hash.replace(/^#\/?/,'')||'architecture';
  state.route=raw==='history'?'publications':['architecture','dashboard','regional','publications'].includes(raw)?raw:'dashboard';
  const atmosphere=$('#page-atmosphere');
  if(atmosphere){
    atmosphere.hidden=!['dashboard','regional','publications'].includes(state.route);
    const nodes=$('.ambient-nodes',atmosphere);
    if(nodes&&!nodes.innerHTML)nodes.innerHTML=Array.from({length:18},(_,i)=>`<i style="--x:${(i*37+11)%100}%;--y:${(i*23+9)%100}%;--delay:${-i*.7}s;--duration:${6+i%5}s"></i>`).join('');
  }
  if(raw!==state.route)history.replaceState(null,'',`#${state.route}`);
  const routeName={architecture:'Accueil',dashboard:'Résultats',regional:'Modèles régionaux',publications:'Publications'};
  if(state.route!=='dashboard')$('#main').nyxDashboard=false;
  if(state.route!=='regional')$('#main').nyxRegional=false;
  $('#breadcrumb').textContent=routeName[state.route];
  $$('[data-nav]').forEach(link=>{const selected=link.dataset.nav===state.route;link.classList.toggle('active',selected);if(selected)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');});
  try{
    if(state.route==='architecture')await renderArchitecture();
    else if(state.route==='dashboard')renderDashboard();
    else if(state.route==='regional'){
      if(!state.boot)state.boot=await api('/api/bootstrap');
      if(!state.regional.day)state.regional.day=state.boot.catalog?.find(item=>item.id==='nyx_regional_cpu')?.parameters?.find(item=>item.name==='delivery_day')?.default||state.runInfo?.defaults?.delivery_day||'';
      renderRegional();
      await Promise.allSettled([checkRegional(),checkRegionalEvaluation(),checkAnnualCwe(),refreshRegional()]);
    }else renderPublications();
  }
  catch(error){$('#main').innerHTML=empty('Chargement impossible',error.message);}
}
async function refresh(){
  if(state.refreshing)return;
  state.refreshing=true;
  const runRevision=state.runRevision;
  try{
    const results=await Promise.allSettled([api('/api/primary-results'),api('/api/primary-run')]);
    let changed=false;
    if(results[0].status==='fulfilled'){
      const next=results[0].value;
      changed=JSON.stringify(next)!==JSON.stringify(state.primary);
      const followedLatest=!state.selectedDay||state.selectedDay===state.primary.days[0]?.date;
      state.primary=next;
      if(followedLatest||!next.days.some(day=>day.date===state.selectedDay))state.selectedDay=next.days[0]?.date||null;
      $('#publication-count').textContent=next.days.length;
      const count=$('#architecture-publication-count');if(count)count.innerHTML=`${next.days.length} <small>↗</small>`;
    }
    // A poll started before POST acceptance cannot erase that accepted run.
    if(runRevision===state.runRevision){
      if(results[1].status==='fulfilled')state.runInfo=results[1].value;
      else state.runInfo=null;
    }
    const ok=results.every(result=>result.status==='fulfilled');connection(ok);
    if(!state.loaded){state.loaded=true;await route();}
    else if(changed&&state.route==='dashboard')renderDashboard();
    else if(changed&&state.route==='publications')renderPublications();
    else renderRunPanel();
    if(results[0].status==='rejected'&&!state.primary.days.length&&state.route==='dashboard')$('#primary-results-content').innerHTML=warnings(['Résultats temporairement indisponibles. Nouvelle tentative automatique.'],'error');
    if(state.route==='regional')await refreshRegional();
  }finally{state.refreshing=false;}
}
document.addEventListener('click',event=>{
  const go=event.target.closest('[data-go]');if(go){location.hash=go.dataset.go;return;}
  const delivery=event.target.closest('[data-delivery]');if(delivery){state.selectedDay=delivery.dataset.delivery;location.hash='dashboard';return;}
  const report=event.target.closest('[data-report]');if(report){showReport(report.dataset.report,report.dataset.title);return;}
  const regionalReport=event.target.closest('[data-regional-report]');if(regionalReport){showRegionalReport(regionalReport.dataset.runId,regionalReport.dataset.regionalReport);return;}
  const regionalRun=event.target.closest('[data-regional-run]');if(regionalRun){state.regional.pinnedRunId=regionalRun.dataset.regionalRun;void refreshRegional();return;}
  if(event.target.closest('[data-close]')){$('#dialog').close();return;}
  if(event.target.closest('[data-launch]'))void launchForecast();
  if(event.target.closest('[data-regional-launch]'))void launchRegional();
  if(event.target.closest('[data-regional-evaluate]'))void launchRegionalEvaluation();
  if(event.target.closest('[data-annual-cwe-check]'))void checkAnnualCwe();
});
document.addEventListener('change',event=>{if(event.target.id==='delivery-day'){state.selectedDay=event.target.value;renderDashboard();}
  if(event.target.id==='regional-day'||event.target.id==='regional-country'){
    state.regional[event.target.id==='regional-day'?'day':'country']=event.target.value;
    state.regional.preflight=null;void checkRegional();
  }
  if(event.target.id==='annual-cwe-country'){
    state.regional.annualCountry=event.target.value;
    state.regional.annualPreflight=null;void checkAnnualCwe();
  }});
document.addEventListener('input',event=>{if(event.target.id==='launch-delivery-day'&&!state.launching&&!state.runInfo?.active)state.launchDay=event.target.value;});
$('#dialog').addEventListener('close',()=>{$('#dialog').innerHTML='';});
window.addEventListener('hashchange',()=>void route());
window.addEventListener('message',handleReportNavigation);
document.addEventListener('visibilitychange',()=>$('#page-atmosphere')?.classList.toggle('suspended',document.hidden));
if(navigator.modelContext?.registerTool){navigator.modelContext.registerTool({name:'list_primary_results',description:'Liste les publications principales NYX par date, sans les expériences.',inputSchema:{type:'object',properties:{},additionalProperties:false},execute:async()=>({content:[{type:'text',text:JSON.stringify(state.primary)}]})});}
void refresh();
setInterval(()=>void refresh(),5000);
