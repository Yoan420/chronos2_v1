/* Interactive, evidence-backed model overview. The animation is conceptual. */
const architectureView = { data: null, view: 'pipeline', selected: 'transformer', motion: true };

function circuitLines() {
  return `<svg class="circuit-connections" viewBox="0 0 960 590" preserveAspectRatio="none" aria-hidden="true">
    <defs><marker id="circuit-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 1L9 5L0 9" fill="none" stroke="currentColor"/></marker></defs>
    <g fill="none" stroke-width="1.4" marker-end="url(#circuit-arrow)">
      <path data-edge="sources pit" d="M127 169 V249"/>
      <path data-edge="pit transformer" d="M230 296 H266 V259 H307"/>
      <path data-edge="transformer residual" d="M595 245 H641 V95 H691"/>
      <path data-edge="residual kalman" d="M805 143 V190"/>
      <path data-edge="kalman forecast" d="M805 284 V331"/>
      <path data-edge="forecast comparison" d="M805 425 V480"/>
      <path data-edge="reference comparison" class="reference-edge" d="M537 519 H662"/>
      <path data-edge="pit residual" class="context-edge" d="M187 250 V43 H751 V47"/>
      <path data-edge="pit kalman" class="context-edge" d="M207 348 V455 H670 V229 H691"/>
    </g>
  </svg>`;
}

function coreArtwork(expanded = false) {
  return `<div class="neural-art ${expanded?'neural-expanded':''}" ${expanded?'role="button" tabindex="0" aria-label="Stimuler le réseau neuronal illustratif"':'aria-hidden="true"'}>
    <canvas class="neural-canvas" aria-hidden="true"></canvas><span class="neural-corner corner-start"></span><span class="neural-corner corner-end"></span>
    ${expanded?'<span class="neural-instruction">Survolez ou cliquez pour envoyer une impulsion</span>':''}</div>`;
}

let neuralArtwork = null;
function stopNeuralArtwork() { neuralArtwork?.destroy(); neuralArtwork = null; }

function startNeuralArtwork() {
  stopNeuralArtwork();
  const canvas = document.querySelector('.neural-canvas');
  if (!canvas) return;
  const host = canvas.parentElement, ctx = canvas.getContext('2d');
  if (!ctx) return;
  const controller = new AbortController(), options = {signal:controller.signal};
  const nodes = [], layers = [], edges = [], pulses = [];
  const colors = ['255,184,103','245,164,104','201,143,181','119,209,215','130,226,222'];
  // Decorative topology, independent of model weights, dimensions and run state.
  [4,6,7,6,4].forEach((count, layer) => {
    layers[layer] = [];
    for(let row=0;row<count;row++) {
      const node = {id:nodes.length,layer,row,x:0,y:0,charge:.15,
        u:.085+layer*.2075,v:.5+(row-(count-1)/2)*.113,z:Math.sin(row*1.7+layer*1.3)};
      nodes.push(node); layers[layer].push(node);
    }
  });
  for(let layer=0;layer<layers.length-1;layer++) {
    for(const source of layers[layer]) {
      for(const target of layers[layer+1]) {
        if(Math.abs(source.v-target.v)<.2 || (source.row+target.row+layer)%5===0)
          edges.push({source,target,lateral:false});
      }
    }
  }
  for(const layer of layers.slice(1,4)) {
    layer.forEach((source,index)=> {if(index<layer.length-1)edges.push({source,target:layer[index+1],lateral:true});});
  }
  let width=1,height=1,frame=0,last=0,elapsed=0,nextWave=0,hover=-1,lastStimulus=-10,inView=true,dead=false;
  let emphasis=-1;
  const moving = () => !dead && canvas.isConnected && inView && !document.hidden && architectureView.motion;
  const rgba=(color,alpha)=>`rgba(${color},${alpha})`;
  function path(edge, position) {
    const a=edge.source,b=edge.target;
    if(!edge.lateral)return {x:a.x+(b.x-a.x)*position,y:a.y+(b.y-a.y)*position};
    return {x:a.x+(b.x-a.x)*position+Math.sin(position*Math.PI)*width*.045,y:a.y+(b.y-a.y)*position};
  }
  function emit(source, strength=1, delay=0, depth=0) {
    if(pulses.length>=120 || depth>5)return;
    edges.filter(e=>e.source===source && (!e.lateral || depth<2)).forEach((edge,index)=>{
      if(pulses.length<120 && (index%2===0 || strength>.9))
        pulses.push({edge,progress:-delay-index*.045,strength,depth});
    });
    source.charge=Math.max(source.charge,strength);
  }
  function stimulate(id=0) {
    if(!architectureView.motion)return;
    emit(nodes[Math.max(0,id)],1.2);
    lastStimulus=elapsed;
  }
  function resize() {
    const box=canvas.getBoundingClientRect();
    width=Math.max(1,box.width); height=Math.max(1,box.height);
    const ratio=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    ctx.setTransform(ratio,0,0,ratio,0,0);draw();
  }
  function draw() {
    ctx.clearRect(0,0,width,height);
    nodes.forEach(n=>{
      n.x=width*(n.u+Math.sin(elapsed*.45+n.z)*.01);
      n.y=height*(n.v+Math.cos(elapsed*.4+n.layer*.8)*.018+n.z*.014);
    });
    const glow=ctx.createRadialGradient(width*.5,height*.5,0,width*.5,height*.5,width*.55);
    glow.addColorStop(0,'rgba(203,116,105,.08)');glow.addColorStop(1,'rgba(16,18,25,0)');
    ctx.fillStyle=glow;ctx.fillRect(0,0,width,height);
    edges.forEach(edge=>{
      const a=edge.source,b=edge.target,connected=hover===a.id||hover===b.id;
      ctx.strokeStyle=rgba(edge.lateral?'202,138,181':colors[a.layer],connected?.6:edge.lateral?.14:.13);
      ctx.lineWidth=connected?1.2:.65;ctx.beginPath();ctx.moveTo(a.x,a.y);
      if(edge.lateral)ctx.quadraticCurveTo(a.x+width*.08,(a.y+b.y)/2,b.x,b.y);
      else ctx.lineTo(b.x,b.y);
      ctx.stroke();
    });
    pulses.forEach(pulse=>{
      if(pulse.progress<0)return;
      const color=colors[pulse.edge.source.layer];
      const point=path(pulse.edge,pulse.progress),tail=path(pulse.edge,Math.max(0,pulse.progress-.19));
      const trail=ctx.createLinearGradient(tail.x,tail.y,point.x+.01,point.y+.01);
      trail.addColorStop(0,rgba(color,0));trail.addColorStop(1,rgba(color,.9));
      ctx.strokeStyle=trail;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(tail.x,tail.y);ctx.lineTo(point.x,point.y);ctx.stroke();
      ctx.fillStyle='#fff0d4';ctx.shadowColor=rgba(color,1);ctx.shadowBlur=8;
      ctx.beginPath();ctx.arc(point.x,point.y,width>350?2:1.3,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
    });
    nodes.forEach(n=>{
      const focus=hover===n.id,boost=Math.min(1,n.charge+(n.layer===emphasis?.22:0));
      const radius=(width>350?3.2:2.1)+boost*1.3;
      ctx.fillStyle=rgba(colors[n.layer],.045+boost*.12);ctx.beginPath();ctx.arc(n.x,n.y,radius*3.7,0,Math.PI*2);ctx.fill();
      ctx.strokeStyle=rgba(colors[n.layer],.35+boost*.5);ctx.lineWidth=.7;
      ctx.beginPath();ctx.arc(n.x,n.y,radius+2.2+boost*2,0,Math.PI*2);ctx.stroke();
      ctx.shadowColor=rgba(colors[n.layer],1);ctx.shadowBlur=boost*15;
      ctx.fillStyle=focus?'#fff3da':rgba(colors[n.layer],.5+boost*.5);
      ctx.beginPath();ctx.arc(n.x,n.y,radius+(focus?1:0),0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
    });
  }
  function tick(timestamp) {
    frame=0;
    if(!moving()) {last=0;return;}
    if(last && timestamp-last<1000/30) {frame=requestAnimationFrame(tick);return;}
    const delta=last?Math.min((timestamp-last)/1000,.075):0;last=timestamp;elapsed+=delta;
    nodes.forEach(n=>n.charge=Math.max(0,n.charge-delta*.95));
    if(elapsed>=nextWave) {
      const first=layers[0][Math.floor(elapsed*.73)%layers[0].length];
      emit(first,1);emit(layers[0][(first.row+2)%layers[0].length],.85,.25);
      nextWave=elapsed+1.75;
    }
    // Process arrivals after iteration so a cascade cannot grow during this frame.
    const arrivals=[];
    for(let i=pulses.length-1;i>=0;i--) {
      const p=pulses[i];p.progress+=delta*(p.edge.lateral?1.25:1.5);
      if(p.progress>=1) {arrivals.push(p);pulses.splice(i,1);}
    }
    arrivals.forEach(p=>{
      const n=p.edge.target;n.charge=Math.max(n.charge,p.strength);
      if(p.strength>.4 && n.layer<4)emit(n,p.strength*.72,.05,p.depth+1);
    });
    draw();frame=requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame);frame=0;last=0;
    host.classList.toggle('neural-paused',!architectureView.motion);
    if(moving())frame=requestAnimationFrame(tick);else draw();
  }
  host.addEventListener('pointermove',event=>{
    const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;
    let closest=-1,distance=width>350?48:30;
    nodes.forEach(n=>{const d=Math.hypot(n.x-x,n.y-y);if(d<distance){closest=n.id;distance=d;}});
    if(closest!==hover) {hover=closest;if(hover>=0 && elapsed-lastStimulus>.4)stimulate(hover);if(!architectureView.motion)draw();}
  },options);
  host.addEventListener('pointerleave',()=>{hover=-1;if(!architectureView.motion)draw();},options);
  host.addEventListener('click',()=>stimulate(hover<0?0:hover),options);
  host.addEventListener('keydown',event=>{
    if((event.key==='Enter'||event.key===' ') && !event.repeat) {event.preventDefault();stimulate(layers[0][Math.floor(elapsed)%4].id);}
  },options);
  document.addEventListener('visibilitychange',sync,options);
  const visibility=new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;sync();});visibility.observe(canvas);
  const sizing=new ResizeObserver(resize);sizing.observe(host);
  resize();sync();
  neuralArtwork={sync,select(id){emphasis=['patches','time_attention','group_attention','feed_forward','quantiles'].indexOf(id);if(emphasis>=0)stimulate(layers[emphasis][0].id);draw();},
    destroy(){dead=true;cancelAnimationFrame(frame);controller.abort();visibility.disconnect();sizing.disconnect();}};
}

function architectureNode(node, index) {
  const special = node.id === 'transformer';
  const short = { sources:'Séries & covariables', pit:'Contrôle point-in-time', transformer:'CHRONOS–2', residual:'Correcteur résiduel', kalman:'Kalman nucléaire', forecast:'Prévisions', reference:'Storm · observations', comparison:'Évaluation & rapports' };
  const caption = { sources:'Prix · nucléaire · exogènes', pit:'Données connues au cutoff', transformer:'ENCODEUR TRANSFORMER', residual:'CatBoost', kalman:'Correction causale', forecast:'Quantiles · day-ahead', reference:'Références indépendantes', comparison:'Model / Storm' };
  return `<button class="architecture-node node-${node.id} ${architectureView.selected === node.id ? 'selected' : ''}" data-arch-node="${node.id}" aria-pressed="${architectureView.selected === node.id}" aria-label="Explorer ${esc(node.title)}">
    <span class="node-code">${String(index+1).padStart(2,'0')} / ${special ? 'FOUNDATION MODEL' : esc(node.kicker || 'MODULE')}</span>
    ${special ? coreArtwork() : ''}<strong>${short[node.id] || esc(node.title)}</strong><span class="node-caption">${caption[node.id] || ''}</span>
    <span class="node-corner" aria-hidden="true">＋</span></button>`;
}

function transformerBoard() {
  const steps = architectureView.data.inner_transformer || [
    {id:'patches',title:'Patches & projection',description:'Les séries sont normalisées et découpées en patches. Valeurs, masque et temps sont projetés en représentations.'},
    {id:'temporal',title:'Attention temporelle',description:'Le modèle relie les positions d’une série pour exploiter son contexte temporel.'},
    {id:'group',title:'Attention de groupe',description:'Les séries appartenant au même groupe échangent de l’information, selon un masque explicite.'},
    {id:'feedforward',title:'Feed-forward',description:'Un réseau transforme les représentations après les deux opérations d’attention.'},
    {id:'quantiles',title:'Projection en quantiles',description:'Les positions futures sont projetées en prévisions probabilistes. Aucun décodeur séparé n’est représenté.'}
  ];
  return `<div class="transformer-deep"><div class="deep-art"><div class="neural-heading"><span>CHRONOS–2 / RÉSEAU NEURONAL</span><strong>Des connexions en mouvement</strong></div>${coreArtwork(true)}</div>
    <div class="transformer-steps">${steps.map((step,i)=>`<button class="inner-step ${architectureView.selected === step.id ? 'selected' : ''}" data-inner-node="${esc(step.id)}"><span>${String(i+1).padStart(2,'0')}</span><strong>${esc(step.title)}</strong><span aria-hidden="true">${i<steps.length-1?'→':'◇'}</span></button>`).join('')}</div>
    <p class="deep-caption">Animation illustrative des échanges. Ces neurones ne représentent ni les dimensions, ni les activations réelles du modèle.</p></div>`;
}

async function renderArchitecture() {
  $('#main').innerHTML = '<div class="loading">Lecture de l’architecture du modèle…</div>';
  const data = await api('/api/architecture');
  if (state.route !== 'architecture') return;
  architectureView.data = data;
  architectureView.selected = 'transformer';
  architectureView.view = 'pipeline';
  architectureView.motion = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  $('#main').innerHTML = `<div class="architecture-hero"><div><div class="eyebrow">CWE</div>
    <h1>Architecture du modèle</h1><p>Des données disponibles au cutoff jusqu’aux résultats.</p></div>
    <div class="cover-access"><span class="cover-model">NYX</span><p>Chronos-2 <span>×</span> CatBoost <span>×</span> Kalman</p><button class="primary" data-go="dashboard">Explorer mes résultats ↗</button></div></div>
    <div class="architecture-toolbar"><div class="architecture-views" role="group" aria-label="Vue du schéma"><button class="selected" data-arch-view="pipeline">Architecture complète</button><button data-arch-view="transformer">Le bloc Transformer</button></div>
      <button class="motion-toggle" data-arch-motion aria-pressed="${architectureView.motion}">${architectureView.motion?'Ⅱ Suspendre':'▷ Animer'} le flux</button></div>
    <div class="architecture-layout"><section class="architecture-board ${architectureView.motion?'':'paused'}" aria-label="Schéma interactif du modèle" id="architecture-board"></section>
      <aside class="architecture-inspector" id="architecture-inspector" aria-live="polite"></aside></div>
    <div class="architecture-footnote"><span><span class="legend-line"></span> Chaîne de prévision <span class="legend-line reference"></span> Références d’évaluation</span><span>Schéma conceptuel · animation indépendante des calculs</span></div>
    <div class="cover-facts">${(data.badges||[]).map(b=>`<div><span>${esc(b.label)}</span><strong>${esc(b.value)}</strong></div>`).join('')}<a href="#publications"><span>LIVRAISONS DISPONIBLES</span><strong id="architecture-publication-count">${state.primary.days.length} <small>↗</small></strong></a></div>
    <details class="architecture-provenance"><summary>Lire le périmètre et les sources du schéma</summary><p>${esc(data.description||'')}</p>${warnings(data.warnings)}<div class="architecture-evidence">${data.nodes.flatMap(n=>(n.evidence||[]).map(e=>`<div><strong>${esc(n.title)}</strong><code>${esc(e.path)}${e.line?':'+e.line:''}</code></div>`)).join('')}</div></details>`;
  renderArchitectureBoard();
  selectArchitectureNode('transformer');
}

function renderArchitectureBoard() {
  stopNeuralArtwork();
  const board = $('#architecture-board');
  if (!board) return;
  const data = architectureView.data;
  board.classList.toggle('deep-mode', architectureView.view === 'transformer');
  board.innerHTML = architectureView.view === 'pipeline'
    ? `<div class="board-coordinate top-coordinate">SYSTEM MAP / 01</div>${circuitLines()}${data.nodes.map(architectureNode).join('')}<span class="board-coordinate bottom-coordinate">SOURCES → MODÈLE → RÉSULTATS</span>`
    : transformerBoard();
  startNeuralArtwork();
}

function selectArchitectureNode(id, inner = false) {
  architectureView.selected = id;
  neuralArtwork?.select(id);
  const data = architectureView.data;
  let node = inner ? (data.inner_transformer || []).find(n=>n.id===id) : data.nodes.find(n=>n.id===id);
  if (!node) node = data.nodes.find(n=>n.id==='transformer');
  $$('[data-arch-node]').forEach(b=>{b.classList.toggle('selected',b.dataset.archNode===id);b.setAttribute('aria-pressed',String(b.dataset.archNode===id));});
  $$('[data-inner-node]').forEach(b=>{b.classList.toggle('selected',b.dataset.innerNode===id);b.setAttribute('aria-pressed',String(b.dataset.innerNode===id));});
  $$('[data-edge]').forEach(e=>e.classList.toggle('highlighted',e.dataset.edge.split(' ').includes(id)));
  $('#architecture-inspector').innerHTML = `<div class="inspector-index">MODULE SÉLECTIONNÉ <span>↙</span></div><div class="inspector-symbol" aria-hidden="true">${id==='transformer'||inner?'◈':id==='kalman'?'∿':id==='forecast'?'⌁':'◇'}</div>
    <div class="eyebrow">${esc(node.kicker || (inner?'CHRONOS–2':'PIPELINE'))}</div><h2>${esc(node.title)}</h2><p>${esc(node.description)}</p>
    ${node.tags?.length?`<div class="node-tags">${node.tags.map(t=>`<span>${esc(t)}</span>`).join('')}</div>`:''}
    ${node.details?.length?`<ul>${node.details.map(d=>`<li>${esc(d)}</li>`).join('')}</ul>`:''}
    ${id==='transformer'&&!inner?'<button class="inspector-link" data-arch-view="transformer">Ouvrir le bloc Transformer →</button>':''}
    <span class="inspector-tip">Cliquez sur un module pour explorer son rôle.</span>`;
}

document.addEventListener('click', e => {
  const node=e.target.closest('[data-arch-node]');
  if(node){selectArchitectureNode(node.dataset.archNode);return;}
  const inner=e.target.closest('[data-inner-node]');
  if(inner){selectArchitectureNode(inner.dataset.innerNode,true);return;}
  const view=e.target.closest('[data-arch-view]');
  if(view){architectureView.view=view.dataset.archView;architectureView.selected='transformer';$$('.architecture-views button').forEach(b=>b.classList.toggle('selected',b.dataset.archView===architectureView.view));renderArchitectureBoard();selectArchitectureNode('transformer');return;}
  const motion=e.target.closest('[data-arch-motion]');
  if(motion){architectureView.motion=!architectureView.motion;$('#architecture-board').classList.toggle('paused',!architectureView.motion);motion.setAttribute('aria-pressed',String(architectureView.motion));motion.textContent=`${architectureView.motion?'Ⅱ Suspendre':'▷ Animer'} le flux`;neuralArtwork?.sync();}
});

window.addEventListener('hashchange',stopNeuralArtwork);
window.addEventListener('pagehide',stopNeuralArtwork);
window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',event=>{
  if(!event.matches)return;
  architectureView.motion=false;
  document.querySelector('#architecture-board')?.classList.add('paused');
  const toggle=document.querySelector('[data-arch-motion]');
  if(toggle){toggle.setAttribute('aria-pressed','false');toggle.textContent='▷ Animer le flux';}
  neuralArtwork?.sync();
});
