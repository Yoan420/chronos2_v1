// Frontend contract tests with a minimal DOM: no browser or scientific execution.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');
const source = fs.readFileSync('experiment_console/static/app.js', 'utf8');

async function main() {
  const elements = new Map();
  let frames = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      innerHTML:'', textContent:'', hidden:false, className:'',
      addEventListener(){}, querySelector:selector=>element(id+' '+selector), classList:{toggle(){}}, setAttribute(){}, removeAttribute(){},
      contains:frame=>Boolean(frame.inDialog),focus(){this.focused=true;},showModal(){this.open=true;},close(){this.open=false;},reportValidity(){return true;},
    });
    return elements.get(id);
  };
  const storage = new Map(), calls = [], intervals = [], listeners = {};
  const file = {path:'reports/model_storm/CWE_Model_Storm_2026-09-12.html',name:'CWE_Model_Storm_2026-09-12.html',updated_at:'2026-09-11T15:00:00Z'};
  let publications = {days:[
    {date:'2026-09-12',cwe:file,zones:[{zone:'FR',label:'France',publication_status:'verified',report:{...file,path:'exports/2026-09-12/fr/nuclear_kalman/forecast_fr_2026-09-12_nuclear_kalman.html'},csv:null,warnings:[]}]},
    {date:'2026-09-09',cwe:null,zones:[]},
  ],warnings:[]};
  let runInfo = {available:true,defaults:{delivery_day:'2026-09-12'},active:null,latest:null};
  let postBehavior = async () => ({run:{id:'new-run',status:'queued'},already_active:false});
  let failRunInfo=false;
  let runInfoGate=null;
  const context = vm.createContext({
    document:{querySelector:element,querySelectorAll:selector=>selector==='iframe[data-cwe-report]'?frames:[],addEventListener(type,callback){(listeners[type]??=[]).push(callback);}},
    window:{addEventListener(){}}, navigator:{},
    location:{hash:'#dashboard'}, history:{replaceState(_a,_b,hash){context.location.hash=hash;}},
    sessionStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    crypto:{randomUUID},
    setTimeout:()=>1,clearTimeout(){},setInterval:callback=>intervals.push(callback),
    renderArchitecture:async()=>{},
    fetch:async(path,options)=>{
      calls.push({path,options});
      if(failRunInfo&&path==='/api/primary-run'&&options.method!=='POST')throw Error('Suivi indisponible');
      if(runInfoGate&&path==='/api/primary-run'&&options.method!=='POST'){
        const pending=runInfoGate;runInfoGate=null;
        const oldInfo=await pending;
        return {ok:true,status:200,json:async()=>oldInfo};
      }
      const value=path==='/api/primary-results'?publications:path==='/api/bootstrap'?{token:'test-token'}:
        options.method==='POST'?await postBehavior():runInfo;
      return {ok:true,status:200,json:async()=>JSON.parse(JSON.stringify(value))};
    },
  });
  vm.runInContext(source,context);
  await new Promise(resolve=>setImmediate(resolve));
  const evaluate = code => vm.runInContext(code,context);
  assert.equal(evaluate('state.selectedDay'),'2026-09-12');
  assert.ok(element('#primary-results-content').innerHTML.includes('Rapport global CWE'));
  assert.equal((element('#primary-results-content').innerHTML.match(/class="primary-zone /g)||[]).length,4);
  assert.ok(element('#primary-run-panel').innerHTML.includes('Lancer la prévision'));
  assert.ok(element('#primary-run-panel').innerHTML.includes('type="date" id="launch-delivery-day"'));
  assert.equal(evaluate('state.launchDay'),'2026-09-12');
  assert.ok(!calls.some(call=>call.options.method==='POST'||call.path==='/api/runs'));
  assert.equal(intervals.length,1);

  // Opaque CWE frames can navigate only to a known delivery. Foreign, removed
  // and country-report frames cannot control the page through postMessage.
  const sent=[];
  const cweFrame={dataset:{deliveryDay:'2026-09-12'},contentWindow:{postMessage:(data,target)=>sent.push({data,target})}};
  frames=[cweFrame];
  function message(source,data){context.reportEvent={source,data};evaluate('handleReportNavigation(reportEvent)');}
  message(cweFrame.contentWindow,{type:'nyx:delivery-ready'});
  assert.equal(sent[0].data.selectedDay,'2026-09-12');
  assert.ok(sent[0].data.dates.includes('2026-09-09'));
  assert.equal(sent[0].target,'*');
  message({}, {type:'nyx:delivery-selected',date:'2026-09-09'});
  message(cweFrame.contentWindow,{type:'nyx:delivery-selected',date:'1999-01-01'});
  assert.equal(evaluate('state.selectedDay'),'2026-09-12');
  message(cweFrame.contentWindow,{type:'nyx:delivery-selected',date:'2026-09-09'});
  assert.equal(evaluate('state.selectedDay'),'2026-09-09');
  assert.equal(evaluate('state.launchDay'),'2026-09-12');
  for(const listener of listeners.input)listener({target:{id:'launch-delivery-day',value:'2026-08-31'}});
  assert.equal(evaluate('state.selectedDay'),'2026-09-09');
  assert.equal(evaluate('state.launchDay'),'2026-08-31');
  const controls=element('#primary-run-panel').innerHTML;
  runInfo.defaults.delivery_day='2026-09-13';
  await evaluate('refresh()');
  assert.equal(evaluate('state.launchDay'),'2026-08-31');
  assert.equal(element('#primary-run-panel').innerHTML,controls,'polling must preserve mounted date control');
  const dashboard=element('#main').innerHTML;
  publications.days[0].zones[0].publication_status='incomplete';
  await evaluate('refresh()');
  assert.equal(element('#main').innerHTML,dashboard,'new publications must not rebuild the dashboard shell');
  assert.equal(element('#primary-run-panel').innerHTML,controls);
  failRunInfo=true;
  await evaluate('refresh()');
  assert.equal(element('#primary-run-panel').innerHTML,controls,'connection errors must not remove the date control');
  assert.equal(element('#launch-forecast').disabled,true);
  failRunInfo=false;
  await evaluate('refresh()');
  assert.equal(evaluate('state.launchDay'),'2026-08-31');
  assert.ok(element('#primary-results-content').innerHTML.includes('Rapport CWE non publié'));
  assert.ok(element('#delivery-day').focused);
  frames=[];
  message(cweFrame.contentWindow,{type:'nyx:delivery-selected',date:'2026-09-12'});
  assert.equal(evaluate('state.selectedDay'),'2026-09-09');
  const modalFrame={...cweFrame,inDialog:true};
  frames=[modalFrame];
  element('#dialog').open=true;
  // A still-open report keeps its own date when the main page selects another.
  message(modalFrame.contentWindow,{type:'nyx:delivery-ready'});
  assert.equal(sent.at(-1).data.selectedDay,'2026-09-12');
  message(modalFrame.contentWindow,{type:'nyx:delivery-selected',date:'2026-09-12'});
  assert.ok(element('#dialog').innerHTML.includes('CWE_Model_Storm_2026-09-12.html'));
  assert.equal(element('#dialog').open,true);
  message(modalFrame.contentWindow,{type:'nyx:delivery-ready'});
  assert.equal(sent.at(-1).data.focus,true);
  message(modalFrame.contentWindow,{type:'nyx:delivery-selected',date:'2026-09-09'});
  assert.equal(element('#dialog').open,false);
  frames=[];

  // Browsing an older delivery never substitutes the latest report or changes
  // the selected delivery on the next polling cycle.
  evaluate("state.selectedDay='2026-09-09'; renderDashboard()");
  assert.ok(element('#primary-results-content').innerHTML.includes('Rapport CWE non publié'));
  assert.ok(!element('#primary-results-content').innerHTML.includes('<iframe'));
  await evaluate('refresh()');
  assert.equal(evaluate('state.selectedDay'),'2026-09-09');
  context.location.hash='#run/old-experiment';
  await evaluate('route()');
  assert.equal(evaluate('state.route'),'dashboard');

  // Available reports are independent from the actual process outcome.
  runInfo.latest={id:'failed-run',status:'failed',finished_at:'2026-09-11T16:00:00Z',error:'Échec du calcul'};
  await evaluate('refresh()');
  assert.ok(element('#run-launch-status').innerHTML.includes('Calcul échoué'));
  assert.ok(element('#run-launch-status').innerHTML.includes('Échec du calcul'));

  runInfo.latest.progress={completed:6,total:6,percent:100,steps:[{name:'sources',label:'Sources',status:'complete'},{name:'nuclear_kalman',zone:'BE',status:'failed'}]};
  await evaluate('refresh()');
  assert.ok(element('#run-launch-status').innerHTML.includes('aria-valuenow="100"'));
  assert.ok(element('#run-launch-status').innerHTML.includes('6 / 6 étapes traitées'));
  assert.ok(element('#run-launch-status').innerHTML.includes('run-progress failed'));
  runInfo.active={status:'running',progress:{percent:null,completed:null,total:null,indeterminate:true}};
  await evaluate('refresh()');
  assert.ok(element('#run-launch-status').innerHTML.includes('run-progress-track indeterminate'));
  assert.ok(!element('#run-launch-status').innerHTML.includes('aria-valuenow'));
  assert.equal(element('#launch-delivery-day').disabled,true);
  const beforeActive=calls.length;
  await evaluate('launchForecast()');
  assert.equal(calls.length,beforeActive);
  runInfo.active=null;
  await evaluate('refresh()');
  assert.equal(element('#launch-delivery-day').disabled,false);
  for(const invalid of ['', '2026-02-29', '0000-01-01', '2026-2-01']){
    context.invalidDay=invalid;
    evaluate('state.launchDay=invalidDay');
    await evaluate('launchForecast()');
  }
  assert.ok(!calls.some(call=>call.options.method==='POST'));
  evaluate("state.launchDay='2026-08-31'");

  // An ambiguous network failure retains one intent; double clicks cannot
  // submit a second request while that same launch is still in flight.
  let rejectPost;
  postBehavior = ()=>new Promise((_resolve,reject)=>{rejectPost=reject;});
  const first = evaluate('launchForecast()');
  await new Promise(resolve=>setImmediate(resolve));
  await evaluate('launchForecast()');
  assert.equal(calls.filter(call=>call.options.method==='POST').length,1);
  const firstKey=calls.find(call=>call.options.method==='POST').options.headers['Idempotency-Key'];
  assert.ok(firstKey);
  rejectPost(new Error('Connexion interrompue'));
  await first;
  assert.deepEqual(JSON.parse(storage.get('nyx-launch-intent')),{key:firstKey,delivery_day:'2026-08-31'});
  postBehavior=async()=>({run:{id:'new-run',status:'queued'},already_active:true});
  failRunInfo=true;
  await evaluate('launchForecast()');
  const posts=calls.filter(call=>call.options.method==='POST');
  assert.equal(posts[1].options.headers['Idempotency-Key'],firstKey);
  assert.equal(posts[1].options.headers['X-Console-Token'],'test-token');
  assert.equal(posts[1].options.body,'{"delivery_day":"2026-08-31"}');
  assert.equal(storage.has('nyx-launch-intent'),false);
  assert.equal(evaluate('state.runInfo.active.id'),'new-run','POST acceptance must immediately lock further launches without a follow-up GET');
  assert.ok(!element('#toast').textContent.includes('indisponible'));
  await evaluate('launchForecast()');
  assert.equal(calls.filter(call=>call.options.method==='POST').length,2);
  failRunInfo=false;
  await evaluate('refresh()');
  // A different date is a different intent, even after a failed request.
  storage.set('nyx-launch-intent',JSON.stringify({key:firstKey,delivery_day:'2026-08-31'}));
  evaluate("state.launchDay='2024-02-29'");
  await evaluate('launchForecast()');
  const leapPost=calls.filter(call=>call.options.method==='POST').at(-1);
  assert.notEqual(leapPost.options.headers['Idempotency-Key'],firstKey);
  assert.equal(leapPost.options.body,'{"delivery_day":"2024-02-29"}');
  assert.equal(evaluate('state.selectedDay'),'2026-09-09');
  // A GET already in flight before acceptance cannot erase the accepted POST.
  await evaluate('refresh()');
  let releasePoll;
  runInfoGate=new Promise(resolve=>{releasePoll=resolve;});
  const stalePoll=evaluate('refresh()');
  await new Promise(resolve=>setImmediate(resolve));
  await evaluate('launchForecast()');
  releasePoll(JSON.parse(JSON.stringify(runInfo)));
  await stalePoll;
  assert.equal(evaluate('state.runInfo.active.id'),'new-run');

  // Published names and warnings cannot inject controls into the main UI.
  publications.days[0].zones[0].label='<img src=x onerror=alert(1)>';
  await evaluate('refresh()');
  evaluate("state.selectedDay='2026-09-12'; renderDashboard()");
  assert.ok(!element('#primary-results-content').innerHTML.includes('<img src=x'));
  assert.ok(element('#primary-results-content').innerHTML.includes('&lt;img'));
  console.log('NYX frontend contracts passed: publications, navigation, failed status, launch intent, double click, escaping.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
