'use strict';
const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {route:'',primary:{days:[],warnings:[]},runInfo:null,runRevision:0,boot:null,selectedDay:null,launchDay:null,refreshing:false,loaded:false,launching:false};
const warnings = (items, kind='') => items?.length ? `<div class="notice ${kind}">${items.map(esc).join('<br>')}</div>` : '';
const formatDay = value => value ? new Date(`${value}T12:00:00`).toLocaleDateString('fr-FR',{day:'numeric',month:'long',year:'numeric'}) : 'Date indisponible';
const formatTime = value => value ? new Date(value).toLocaleString('fr-FR',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}) : 'Non disponible';
const artifactUrl = (file, download=false) => `/api/primary-artifact?path=${encodeURIComponent(file.path)}${download?'&download=1':''}`;
const statusLabels = {queued:'En attente',starting:'Démarrage',running:'Calcul en cours',succeeded:'Calcul terminé',failed:'Calcul échoué',cancelling:'Arrêt demandé',cancelled:'Calcul arrêté',interrupted:'Calcul interrompu'};
function toast(message){const target=$('#toast');target.textContent=message;target.hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>target.hidden=true,8000);}
async function api(path, body, retry=true, extraHeaders={}){
  const options={cache:'no-store'};
  if(body!==undefined){options.method='POST';options.headers={'Content-Type':'application/json','X-Console-Token':state.boot?.token||'',...extraHeaders};options.body=JSON.stringify(body);}
  const response=await fetch(path,options);
  let value;
  try{value=await response.json();}catch{throw Error('Réponse illisible du serveur local.');}
  if(response.status===403&&body!==undefined&&retry&&value.error?.startsWith('Session expirée')){state.boot=await api('/api/bootstrap');return api(path,body,false,extraHeaders);}
  if(!response.ok)throw Error(value.error||`Erreur ${response.status}`);
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
async function route(){
  const raw=location.hash.replace(/^#\/?/,'')||'architecture';
  state.route=raw==='history'?'publications':['architecture','dashboard','publications'].includes(raw)?raw:'dashboard';
  const atmosphere=$('#page-atmosphere');
  if(atmosphere){
    atmosphere.hidden=!['dashboard','publications'].includes(state.route);
    const nodes=$('.ambient-nodes',atmosphere);
    if(nodes&&!nodes.innerHTML)nodes.innerHTML=Array.from({length:18},(_,i)=>`<i style="--x:${(i*37+11)%100}%;--y:${(i*23+9)%100}%;--delay:${-i*.7}s;--duration:${6+i%5}s"></i>`).join('');
  }
  if(raw!==state.route)history.replaceState(null,'',`#${state.route}`);
  const routeName={architecture:'Accueil',dashboard:'Résultats',publications:'Publications'};
  if(state.route!=='dashboard')$('#main').nyxDashboard=false;
  $('#breadcrumb').textContent=routeName[state.route];
  $$('[data-nav]').forEach(link=>{const selected=link.dataset.nav===state.route;link.classList.toggle('active',selected);if(selected)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');});
  try{if(state.route==='architecture')await renderArchitecture();else if(state.route==='dashboard')renderDashboard();else renderPublications();}
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
  }finally{state.refreshing=false;}
}
document.addEventListener('click',event=>{
  const go=event.target.closest('[data-go]');if(go){location.hash=go.dataset.go;return;}
  const delivery=event.target.closest('[data-delivery]');if(delivery){state.selectedDay=delivery.dataset.delivery;location.hash='dashboard';return;}
  const report=event.target.closest('[data-report]');if(report){showReport(report.dataset.report,report.dataset.title);return;}
  if(event.target.closest('[data-close]')){$('#dialog').close();return;}
  if(event.target.closest('[data-launch]'))void launchForecast();
});
document.addEventListener('change',event=>{if(event.target.id==='delivery-day'){state.selectedDay=event.target.value;renderDashboard();}});
document.addEventListener('input',event=>{if(event.target.id==='launch-delivery-day'&&!state.launching&&!state.runInfo?.active)state.launchDay=event.target.value;});
$('#dialog').addEventListener('close',()=>{$('#dialog').innerHTML='';});
window.addEventListener('hashchange',()=>void route());
window.addEventListener('message',handleReportNavigation);
document.addEventListener('visibilitychange',()=>$('#page-atmosphere')?.classList.toggle('suspended',document.hidden));
if(navigator.modelContext?.registerTool){navigator.modelContext.registerTool({name:'list_primary_results',description:'Liste les publications principales NYX par date, sans les expériences.',inputSchema:{type:'object',properties:{},additionalProperties:false},execute:async()=>({content:[{type:'text',text:JSON.stringify(state.primary)}]})});}
void refresh();
setInterval(()=>void refresh(),5000);
