
/* ================================================================
   recap-comic studio — app core
   Talks ONLY to the existing FastAPI routes in webapp/main.py.
   ================================================================ */
"use strict";
const $  = (s,el=document)=>el.querySelector(s);
const $$ = (s,el=document)=>[...el.querySelectorAll(s)];
const esc = s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = s=>{s=Math.max(0,s||0);return `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,"0")}`};
const ago = t=>{if(!t)return"";const d=(Date.now()/1000-t)|0;return d<60?`${d}s ago`:d<3600?`${d/60|0}m ago`:d<86400?`${d/3600|0}h ago`:`${d/86400|0}d ago`};
const fileURL=(session,name)=>`/api/jobs/${session}/files/${encodeURIComponent(name)}`;
/* ---------------- API client (existing routes only) ---------------- */
async function req(path,opt={}){
  const r=await fetch(path,opt);
  if(!r.ok){let m=r.statusText;try{m=(await r.json()).detail||m}catch(_){}throw new Error(m)}
  const ct=r.headers.get("content-type")||"";
  return ct.includes("json")?r.json():r;
}
const jbody=b=>({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});
const api={
  config:      ()=>req("/api/config"),
  projects:    ()=>req("/api/projects"),
  upload:      (f,run=true)=>{const fd=new FormData();fd.append("file",f);return req(`/api/upload?run=${run?1:0}`,{method:"POST",body:fd})},
  run:         b=>req("/api/run",jbody(b)),
  job:         id=>req(`/api/jobs/${id}`),
  jobLogs:     id=>req(`/api/jobs/${id}?logs=1`),
  cancel:      id=>req(`/api/jobs/${id}/cancel`,{method:"POST"}),
  edGet:       s=>req(`/api/editor/${s}`),
  edCreate:    s=>req(`/api/editor/${s}/create`,{method:"POST"}),
  edPost:      (s,what,body={})=>req(`/api/editor/${s}/${what}`,jbody(body)),
  panelsGet:   s=>req(`/api/panels/${s}`),
  panelsOrder: (s,ids)=>req(`/api/panels/${s}/order`,jbody({ids})),
  panelsDelete:(s,ids)=>req(`/api/panels/${s}/delete`,jbody({ids})),
  panelsRestore:(s,ids)=>req(`/api/panels/${s}/restore`,jbody({ids})),
  panelsReview:(s,pid,st)=>req(`/api/panels/${s}/review`,jbody({panel_id:pid,status:st})),
  panelsConfirm:(s,reviewAll=true)=>req(`/api/panels/${s}/confirm`,jbody({review_all:reviewAll})),
  panelsDuplicate:(s,pid)=>req(`/api/panels/${s}/duplicate`,jbody({panel_id:pid})),
  panelsMerge:   (s,ids)=>req(`/api/panels/${s}/merge`,jbody({ids})),
  panelsSplit:   (s,pid,frac)=>req(`/api/panels/${s}/split`,jbody({panel_id:pid,fraction:frac})),
  narrGet:       s=>req(`/api/narration/${s}`),
  narrText:      (s,pid,text)=>req(`/api/narration/${s}/text`,jbody({panel_id:pid,text})),
  narrReset:     (s,pid)=>req(`/api/narration/${s}/reset`,jbody({panel_id:pid})),
  narrStyle:     (s,style)=>req(`/api/narration/${s}/style`,jbody({style})),
  narrRegen:     (s,pid)=>req(`/api/narration/${s}/regenerate`,jbody({panel_id:pid,api_key:store.settings.api_key,model:store.settings.model})),
  voiceGet:    s=>req(`/api/voice/${s}`),
  voicePut:    (s,cfg)=>req(`/api/voice/${s}`,jbody(cfg)),
  voiceList:   p=>req(`/api/voices?provider=${encodeURIComponent(p||"edge")}`),
  voicePreview:(s,cfg,text)=>req(`/api/voice/${s}/preview`,jbody({cfg,text})),
};
/* ---------------- store (persisted) ---------------- */
const store=Object.assign({
  session:null, recent:[], order:{}, genJob:{}, names:{}, queue:[],
  settings:{backend:"gemini",model:"",tts:"edge",voice:"en-US-AriaNeural",style:"recap",
            api_key:"",endpoint:"",cf_account_id:""}
}, JSON.parse(localStorage.getItem("rc.studio.v1")||"{}"));
const save=()=>localStorage.setItem("rc.studio.v1",JSON.stringify(store));
/* pipeline stages — the real spine from webapp/pipeline.py (generate) */
const STAGES=[
  ["validate_config","Check config"],["load_images","Load images"],
  ["segment_panels","Segment panels"],["apply_order","Apply order"],
  ["gemini_narration","Gemini narration"],["build_script","Build script"],
  ["render_video","Render video"],["save_outputs","Save outputs"],
  ["create_editor_project","Editor project"]
];
const TERMINAL=["completed","failed","cancelled"];
/* ---------------- toasts / status pill ---------------- */
function toast(msg,kind="ok"){
  const t=document.createElement("div");t.className=`toast ${kind}`;t.textContent=msg;
  $("#toasts").append(t);
  setTimeout(()=>{t.classList.add("out");setTimeout(()=>t.remove(),260)},3600);
}
function setPill(status,txt){
  $("#statusPill").dataset.status=status;
  $("#statusTxt").textContent=txt||status;
}
/* ---------------- job watching (real progress, no fakery) ---------------- */
const watchers=new Map();
function watch(jobId,onChange){
  if(watchers.has(jobId))return;
  const poll=async()=>{
    try{
      const job=await api.job(jobId);
      setPill(job.status, job.status==="running" ? (job.stage||"running").replace(/_/g," ") : job.status);
      onChange&&onChange(job);
      if(TERMINAL.includes(job.status)){
        clearInterval(watchers.get(jobId));watchers.delete(jobId);
        if(job.status==="completed")toast(`Job ${jobId.slice(0,6)} completed`);
        if(job.status==="failed")toast(job.error||"Job failed","err");
      }
    }catch(_){/* transient; keep polling */}
  };
  watchers.set(jobId,setInterval(poll,1000));poll();
}
/* ---------------- router ---------------- */
const viewEl=$("#view");
const routes={studio:vStudio,panels:vPanels,narrate:vNarrate,voice:vVoice,generate:vGenerate,review:vReview,editor:vEditor,logs:vLogs,settings:vSettings};
function parseHash(){const h=location.hash.replace(/^#\/?/,"");const[a,b]=h.split("/");return{view:a||"studio",arg:b}}
function nav(view,arg){location.hash=arg?`#/${view}/${arg}`:`#/${view}`}
async function render(){
  const{view,arg}=parseHash();
  $$(".rail a").forEach(a=>a.classList.toggle("on",a.dataset.view===view));
  viewEl.classList.remove("vin");void viewEl.offsetWidth;
  try{await(routes[view]||vStudio)(arg)}
  catch(e){viewEl.innerHTML=`<div class="panel"><div class="lbl">Error</div><p style="margin-top:8px">${esc(e.message)}</p></div>`}
  viewEl.classList.add("vin");
  bindReveals();
}
addEventListener("hashchange",render);
function bindReveals(){
  const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target)}}),{threshold:.08});
  $$(".reveal",viewEl).forEach(el=>io.observe(el));
}
/* ================================================================
   VIEW — Studio (opens with the strip, not a hero)
   ================================================================ */
async function vStudio(){
  let projects=[];
  try{const r=await api.projects();projects=(r.projects||[]).slice(0,8)}catch(_){}
  const rc=projects.length?projects:(store.recent.slice(0,6).map(r=>({id:r.session,status:r.status||"idle",created_at:r.created,panels:0})));
  viewEl.innerHTML=`
  <section class="studio">
    <div>
      <div class="lbl">manhwa recap studio</div>
      <h1 class="h-disp">One tall strip in.<br><span class="go">One narrated recap out.</span></h1>
      <p class="lede">Automation segments the strip, Gemini narrates it, TTS voices it, FFmpeg renders a 9:16 recap — then you fix only what you don't like in the editor.</p>
      <div class="dropzone reveal" id="dz" role="button" tabindex="0" aria-label="Drop strip images or click to choose">
        <div class="stripmark" aria-hidden="true"></div>
        <div>
          <h3>Drop strips here</h3>
          <p>PNG · JPG · WebP — add one or more full-height chapter strips. Nothing runs until you press Run.</p>
          <div class="fmts"><span class="chip">multiple strips</span><span class="chip">gemini narration</span><span class="chip">edge-tts</span><span class="chip">9:16 render</span></div>
        </div>
      </div>
      <input type="file" id="fileIn" accept="image/png,image/jpeg,image/webp" multiple hidden>
      ${store.queue.length?`
      <div class="panel lift reveal" style="--d:40ms;margin-top:18px">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
          <div class="lbl">ready to run · ${store.queue.length} queued</div>
          <button class="btn sm acc" id="runAll">Run all (${store.queue.length})</button>
        </div>
        <div class="recent" style="margin-top:12px">
          ${store.queue.map(q=>`
            <article class="rcard" data-q="${esc(q.session)}" style="cursor:default">
              <div style="min-width:0">
                <div class="name">${esc(store.names?.[q.session]||q.session)}</div>
                <div class="sub">${esc(q.session)} · <span style="color:var(--amber)">queued</span></div>
              </div>
              <span class="pill" data-status="idle"><span class="dot"></span>queued</span>
              <div class="acts">
                <button class="btn sm danger" data-qdel="${esc(q.session)}">Remove</button>
                <button class="btn sm acc" data-qrun="${esc(q.session)}">Run</button>
              </div>
            </article>`).join("")}
        </div>
      </div>`:""}
    </div>
    <aside class="sysgrid">
      <div class="panel lift reveal" style="--d:80ms">
        <div class="lbl">System</div>
        <div class="statrow" style="margin-top:12px" id="sysStats">
          <div class="stat"><div class="num grn" id="stSess">${projects.length?projects.length:store.recent.length}</div><div class="lbl" style="margin-top:4px">sessions</div></div>
          <div class="stat"><div class="num" id="stG">…</div><div class="lbl" style="margin-top:4px">gemini</div></div>
        </div>
        <p style="margin-top:12px;font-size:12.5px;color:var(--txt3)" id="sysModel"></p>
      </div>
      <div class="panel lift reveal" style="--d:160ms">
        <div class="lbl">Recent sessions</div>
        <div class="recent" style="margin-top:12px">
          ${rc.length?rc.map(p=>`
            <article class="rcard" data-s="${esc(p.id)}">
              <div style="min-width:0">
                <div class="name">${esc(store.names?.[p.id]||p.id)}</div>
                <div class="sub">${esc(p.id)} · ${p.panels||0} panels · ${ago(p.updated_at||p.created_at)}</div>
              </div>
              <span class="pill" data-status="${esc(p.status==="no_job"?"idle":(p.status||"idle"))}"><span class="dot"></span>${esc(p.status==="no_job"?"idle":(p.status||"idle"))}</span>
              <div class="acts">
                <button class="btn sm" data-go="panels">Panels</button>
                ${p.has_video?`<button class="btn sm" data-go="review">Review</button>`:""}
                <button class="btn sm acc" data-go="editor">Editor</button>
              </div>
            </article>`).join("")
          :`<p style="color:var(--txt3);font-size:13px">Nothing yet — drop a strip to create your first session.</p>`}
        </div>
      </div>
    </aside>
  </section>`;
  /* system status from the real /api/config */
  api.config().then(c=>{
    $("#stG").textContent=c.gemini_configured?"✓":"—";
    $("#stG").style.color=c.gemini_configured?"var(--acc)":"var(--coral)";
    $("#sysModel").textContent=c.gemini_configured?`Active model: ${c.model}`:"No Gemini key configured — add one in Settings or .env.";
    const b=$("#gBadge");b.textContent=`gemini: ${c.gemini_configured?c.model:"not set"}`;b.classList.toggle("ok",!!c.gemini_configured);
  }).catch(()=>{});
  /* dropzone wiring */
  const dz=$("#dz"),fi=$("#fileIn");
  dz.addEventListener("click",()=>fi.click());
  dz.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();fi.click()}});
  ["dragover","dragenter"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add("over")}));
  ["dragleave","drop"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove("over")}));
  dz.addEventListener("drop",e=>{const fs=[...e.dataTransfer.files];if(fs.length)queueFiles(fs)});
  fi.addEventListener("change",()=>{if(!fi.files.length)return;const fs=[...fi.files];fi.value="";queueFiles(fs)});
  if($("#runAll"))$("#runAll").onclick=()=>runAllQueued();
  $$("[data-qrun]",viewEl).forEach(b=>b.onclick=e=>{e.stopPropagation();runQueued(b.dataset.qrun)});
  $$("[data-qdel]",viewEl).forEach(b=>b.onclick=e=>{e.stopPropagation();dropQueued(b.dataset.qdel)});
  $$(".rcard[data-s]",viewEl).forEach(card=>{
    const s=card.dataset.s;
    card.addEventListener("click",e=>{
      const go=e.target.closest("[data-go]")?.dataset.go;
      setSession(s); nav(go||"panels", s);
    });
  });
}
/* Upload one or more strips WITHOUT auto-running; they land in the queue
   until the user presses Run. */
async function queueFiles(files){
  const accepted=[...files].filter(f=>/\.(png|jpe?g|webp)$/i.test(f.name));
  if(!accepted.length){toast("Only PNG / JPG / WebP images are supported","err");return}
  toast(`Uploading ${accepted.length} strip${accepted.length>1?"s":""}…`);
  let ok=0,failed=0;
  for(const f of accepted){
    try{
      const{job_id}=await api.upload(f,false);     /* run=0 — no auto-segmentation */
      setSession(job_id,f.name);
      if(!store.queue.some(q=>q.session===job_id)){
        store.queue.unshift({session:job_id,name:f.name,added:Date.now()/1000});
        store.queue=store.queue.slice(0,20);save();
      }
      ok++;
    }catch(e){failed++;toast(`${f.name}: ${e.message}`,"err")}
  }
  if(ok){toast(`Added ${ok} strip${ok>1?"s":""} — press Run when ready`,"ok");render();}
  else if(!failed){toast("Nothing to add","warn")}
}
function runBody(session){
  return{session,order:null,
    tts:store.settings.tts,voice:store.settings.voice,style:store.settings.style,
    backend:store.settings.backend,api_key:store.settings.api_key,
    model:store.settings.model,endpoint:store.settings.endpoint,
    cf_account_id:store.settings.cf_account_id};
}
async function runQueued(session){
  try{
    toast("Automation started — segmenting","ok");
    const{job_id}=await api.run(runBody(session));
    store.genJob[session]=job_id;
    store.queue=store.queue.filter(q=>q.session!==session);save();
    watch(job_id);
    nav("generate",session);
  }catch(e){toast(e.message,"err")}
}
async function runAllQueued(){
  const snap=[...store.queue];
  if(!snap.length){toast("Queue is empty","warn");return}
  let ok=0,failed=0;
  for(const q of snap){
    try{
      const{job_id}=await api.run(runBody(q.session));
      store.genJob[q.session]=job_id;watch(job_id);ok++;
    }catch(e){failed++;toast(`${q.session.slice(0,6)}: ${e.message}`,"err")}
  }
  store.queue=[];save();
  if(ok)toast(`Started automation on ${ok} session${ok>1?"s":""}`,"ok");
  render();
}
function dropQueued(session){
  store.queue=store.queue.filter(q=>q.session!==session);save();
  toast("Removed from queue","warn");
  render();
}
function setSession(s,name){
  store.session=s;if(name)store.names=Object.assign(store.names||{},{[s]:name});save();
  $("#projBtn").textContent=`${store.names?.[s]||s}`;
}
/* Resolve the session a view should act on. Bounces are painful, so when
   nothing is open we auto-select the most recently touched project instead. */
async function requireSession(hint){
  if(hint)return hint;
  if(store.session)return store.session;
  try{
    const r=await api.projects();
    const list=(r.projects||[]).filter(p=>p.id)
      .sort((a,b)=>(b.updated_at||b.created_at||0)-(a.updated_at||a.created_at||0));
    if(list.length){
      setSession(list[0].id);
      return store.session;
    }
  }catch(_){}
  return null;
}
/* Soft landing when a session-requiring view has no session at all. */
function studioEmpty(view){
  viewEl.innerHTML=`<div class="panel lift" style="max-width:520px;margin:56px auto">
    <div class="lbl">${view}</div>
    <h2 class="h-disp" style="margin-top:6px;font-size:26px">No session open</h2>
    <p style="margin-top:10px;color:var(--txt3);font-size:13.5px">This view works on a session. Drop a strip in Studio — segmentation starts immediately — or pick one from the recent list.</p>
    <div style="display:flex;gap:10px;margin-top:14px">
      <button class="btn acc" onclick="location.hash='#/studio'">Go to Studio</button>
    </div>
  </div>`;
}
/* ================================================================
   VIEW — Panels (tall frames, never cropped; drag + keyboard reorder)
   Note: reads panels.json (the authoritative, narration-rich list that
   every run rewrites) and falls back to the in-memory job.panels.
   ================================================================ */
async function vPanels(session){
  session=await requireSession(session);
  if(!session)return studioEmpty("Panels");
  setSession(session);
  let R;
  try{R=await api.panelsGet(session)}
  catch(e){viewEl.innerHTML=`<div class="panel"><div class="lbl">panel review</div>
    <h2 class="h-disp" style="font-size:26px;margin:8px 0">No panels yet</h2>
    <p style="color:var(--txt3);font-size:13.5px">${esc(e.message)}</p>
    <div style="display:flex;gap:10px;margin-top:14px">
      <button class="btn acc" onclick="location.hash='#/studio'">Go to Studio</button>
      <button class="btn" onclick="location.hash='#/generate/${esc(session)}'">Check automation</button>
    </div></div>`;return}
  const act=()=>R.panels.filter(p=>!p.deleted);
  const del=()=>R.panels.filter(p=>p.deleted);
  const need=()=>act().filter(p=>p.review_status==="needs_review");
  viewEl.innerHTML=`
  <div style="display:flex;align-items:end;justify-content:space-between;gap:14px;flex-wrap:wrap">
    <div>
      <div class="lbl">panel review · ${esc(session)}</div>
      <h2 class="h-disp" style="font-size:30px;margin-top:4px">${R.active} panels
        ${R.confirmed?`<span class="chip ok" style="vertical-align:middle;margin-left:8px">✓ confirmed</span>`
          :need().length?`<span class="chip warn" style="vertical-align:middle;margin-left:8px">${need().length} need review</span>`:""}</h2>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button class="btn" id="revNext" ${need().length?"":"disabled"} title="Jump to the lowest-confidence unreviewed panel">Review Next</button>
      <button class="btn acc" id="confirmBtn">Confirm Panels &amp; Continue</button>
    </div>
  </div>
  <p style="color:var(--txt3);font-size:12.5px;margin:8px 0 18px">Verify the AI's detection <b style="color:var(--txt2)">before narration is generated</b>. Drag to reorder (or focus + Space, then ←/→); click a panel to inspect, delete, duplicate, merge or split it. Every change autosaves — the AI original is never overwritten, and deleted panels stay recoverable.</p>
  <div class="pgrid" id="pg">${act().length?act().map((p,i)=>panelCard(p,i)).join("")
    :`<div class="panel" style="text-align:center"><p style="color:var(--txt3);font-size:13px">No active panels — restore some below, or re-run segmentation.</p></div>`}</div>
  ${del().length?`<hr class="cutline"><div class="lbl">deleted · ${del().length} (recoverable)</div>
  <div class="pgrid" style="margin-top:12px">${del().map(p=>`
    <article class="panel pcard" style="opacity:.45" data-del="${esc(p.id)}">
      <div class="frame"><img loading="lazy" src="${fileURL(session,p.image_file)}" alt=""></div>
      <div class="meta" style="flex-direction:row;align-items:center;justify-content:space-between">
        <span class="chip bad">deleted</span>
        <button class="btn sm" data-restore="${esc(p.id)}">↺ Restore</button>
      </div></article>`).join("")}</div>`:""}`;
  const grid=$("#pg");
  const refresh=()=>vPanels(session);
  /* order commit -> persisted server-side (panels_edit.json order) */
  const commitOrder=async()=>{
    const ids=$$(".pcard",grid).map(c=>c.dataset.id);
    if(ids.length<2)return;
    try{await api.panelsOrder(session,ids);toast("Order saved","ok")}
    catch(e){toast(e.message,"err");refresh()}
  };
  wirePanelGrid(session,act(),openReview,commitOrder);
  $$("[data-restore]",viewEl).forEach(b=>b.onclick=async ev=>{
    ev.stopPropagation();
    try{await api.panelsRestore(session,[b.dataset.restore]);toast("Panel restored");refresh()}
    catch(e){toast(e.message,"err")}
  });
  $("#revNext").onclick=()=>{
    const worst=need().sort((a,b)=>(a.confidence??1)-(b.confidence??1))[0];
    if(worst)openReview(worst,act());
  };
  $("#confirmBtn").onclick=()=>{
    const a=act(),low=a.filter(p=>(p.confidence??0)<0.6);
    $("#cfBody").innerHTML=`
      <div>${a.length} panels in final order</div>
      <div style="color:var(--acc)">✓ ${a.length-low.length} high confidence</div>
      ${low.length?`<div style="color:var(--amber)">⚠ ${low.length} low confidence</div>`:""}
      ${del().length?`<div style="color:var(--coral)">${del().length} deleted (recoverable)</div>`:""}
      ${need().length?`<div style="color:var(--amber)">${need().length} still marked needs-review (you can continue anyway)</div>`:""}
      <div style="margin-top:6px;color:var(--txt3)">Next: generate narration for these ${a.length} panels, in this exact order.</div>`;
    $("#confirmDlg").showModal();
  };
  $("#cfBack").onclick=()=>$("#confirmDlg").close();
  $("#cfGo").onclick=async()=>{
    try{await api.panelsConfirm(session,true);$("#confirmDlg").close();toast("Panels confirmed");nav("narrate",session)}
    catch(e){toast(e.message,"err")}
  };
}
function panelCard(p,i){
  const low=(p.confidence??1)<0.6;
  const st=p.review_status||"needs_review";
  const stChip=st==="reviewed"?`<span class="chip ok">✓ reviewed</span>`
    :st==="edited"?`<span class="chip info">✎ edited</span>`
    :`<span class="chip ${low?"warn":""}">· needs review</span>`;
  return`
  <article class="panel lift pcard reveal" draggable="true" data-id="${esc(p.id)}" data-i="${i}"
           tabindex="0" aria-label="Panel ${p.display_order}, ${p.panel_type}, confidence ${((p.confidence??0)*100).toFixed(0)} percent" style="--d:${Math.min(i*35,420)}ms">
    <div class="frame"><img loading="lazy" src="${fileURL(store.session,p.image_file)}" alt="Panel ${p.display_order}"></div>
    <span class="ord">${p.display_order}</span>
    <div class="meta">
      <div class="chips">
        <span class="chip ${low?"warn":""}">✓${((p.confidence??0)*100).toFixed(0)}%</span>
        <span class="chip">${esc(p.panel_type)}</span>
      </div>
      <p class="nar">${esc(p.narration)||"<i style='color:var(--txt3)'>no narration yet</i>"}</p>
      <div class="flags"><span class="${p.dialogue?"ok":""}">${p.dialogue?"✓":"○"} dialogue</span></div>
      <div class="chips">${stChip}</div>
    </div>
  </article>`;
}
function wirePanelGrid(session,panels,openInspect,commitOrder){
  const grid=$("#pg");let dragId=null,grabbed=null;
  const cards=()=>$$(".pcard",grid);
  grid.addEventListener("dragstart",e=>{const c=e.target.closest(".pcard");if(!c)return;dragId=c.dataset.id;c.classList.add("dragging")});
  grid.addEventListener("dragend",e=>{e.target.closest(".pcard")?.classList.remove("dragging");commitOrder()});
  grid.addEventListener("dragover",e=>{
    e.preventDefault();
    const c=e.target.closest(".pcard");if(!c||c.dataset.id===dragId)return;
    $$(".pcard",grid).forEach(x=>x.classList.remove("drag-over"));c.classList.add("drag-over");
    const from=cards().find(x=>x.dataset.id===dragId);if(!from)return;
    const r=c.getBoundingClientRect(),after=e.clientY>r.top+r.height/2;
    grid.insertBefore(from,after?c.nextSibling:c);
  });
  grid.addEventListener("click",e=>{
    const c=e.target.closest(".pcard");if(!c)return;
    openInspect(panels.find(p=>p.id===c.dataset.id),panels);
  });
  grid.addEventListener("keydown",e=>{
    const c=e.target.closest(".pcard");if(!c)return;
    if(e.key===" "){e.preventDefault();grabbed=grabbed===c?null:c;c.style.outline=grabbed?"2px dashed var(--amber)":"";return}
    if(!grabbed)return;
    if(e.key==="ArrowRight"||e.key==="ArrowLeft"){
      e.preventDefault();
      const n=e.key==="ArrowRight"?c.nextElementSibling:c.previousElementSibling;
      if(!n)return;
      e.key==="ArrowRight"?grid.insertBefore(c,n.nextSibling):grid.insertBefore(c,n);
      c.focus();commitOrder();
    }
  });
}
async function openReview(p,act){
  if(!p)return;
  const session=store.session;
  const idx=act.findIndex(x=>x.id===p.id);
  const next=act[idx+1],prev=act[idx-1];
  $("#lbImg").src=fileURL(session,p.image_file);
  $("#lbTitle").textContent=`Panel ${p.display_order} · ${p.id}`;
  const conf=(p.confidence??0)*100;
  $("#lbKv").innerHTML=`
    <div>type&nbsp;&nbsp;&nbsp;: ${esc(p.panel_type)}</div>
    <div>confidence: ${conf.toFixed(0)}% ${conf<60?"⚠ low":""}</div>
    <div>order&nbsp;&nbsp;&nbsp;: ${p.display_order} of ${act.length}</div>
    <div>source&nbsp;&nbsp;: Y = ${p.y_start} → ${p.y_end} px (${(p.y_end??0)-(p.y_start??0)} px tall)</div>
    <div>status&nbsp;&nbsp;: ${esc(p.review_status||"needs_review")}</div>
    <div>file&nbsp;&nbsp;&nbsp;&nbsp;: ${esc(p.image_file)}</div>`;
  $("#lbNar").textContent=p.narration||"—";
  $("#lbDlg").textContent=p.dialogue||"—";
  /* actions — every one maps to a real API route */
  const A=[];
  if((p.review_status||"needs_review")==="needs_review")A.push(["keep","✓ Mark reviewed","acc"]);
  A.push(["dup","⧉ Duplicate",""],["split","✂ Split…",""]);
  if(prev)A.push(["mvL","← Move earlier",""]);
  if(next)A.push(["mvR","Move later →",""]);
  if(next)A.push(["merge","⇥ Merge with next",""]);
  A.push(["del","✕ Delete","danger"]);
  $("#lbActs").innerHTML=A.map(([a,t,c])=>`<button class="btn sm ${c}" data-a="${a}">${t}</button>`).join("");
  $("#lbSplit").style.display="none";
  $("#lbActs").onclick=async e=>{
    const b=e.target.closest("[data-a]");if(!b)return;
    const a=b.dataset.a;e.stopPropagation();
    try{
      if(a==="keep"){await api.panelsReview(session,p.id,"reviewed")}
      else if(a==="dup"){await api.panelsDuplicate(session,p.id);toast("Panel duplicated — new unique id")}
      else if(a==="del"){
        if(!confirm(`Delete panel ${p.display_order}? You can restore it later.`))return;
        await api.panelsDelete(session,[p.id]);toast("Panel deleted (recoverable)","warn");
      }
      else if(a==="mvL"||a==="mvR"){
        const ids=act.map(x=>x.id);
        const j=ids.indexOf(p.id),k=a==="mvR"?j+1:j-1;
        [ids[j],ids[k]]=[ids[k],ids[j]];
        await api.panelsOrder(session,ids);
      }
      else if(a==="merge"){
        if(!confirm(`Merge panel ${p.display_order} and ${next.display_order} into one panel (image + dialogue)?`))return;
        await api.panelsMerge(session,[p.id,next.id]);toast("Panels merged");
      }
      else if(a==="split"){$("#lbSplit").style.display="block";return}
      $("#lightbox").close();vPanels(session);
    }catch(err){toast(err.message,"err")}
  };
  $("#lbSplitBtn").onclick=async()=>{
    try{
      const frac=(+$("#lbSplitAt").value)/100;
      await api.panelsSplit(session,p.id,frac);
      toast("Panel split into two");$("#lightbox").close();vPanels(session);
    }catch(err){toast(err.message,"err")}
  };
  $("#lightbox").showModal();
}
$("#lbClose").onclick=()=>$("#lightbox").close();
$("#lightbox").addEventListener("click",e=>{if(e.target.id==="lightbox")e.target.close()});
/* ================================================================
   VIEW — Narration Studio (per-panel script review & editing)
   ================================================================ */
let NR=null; /* {session, data, sel, saveT} */
async function vNarrate(session){
  session=await requireSession(session);if(!session)return studioEmpty("Narration");
  setSession(session);
  try{NR={session,data:await api.narrGet(session),sel:null,saveT:null}}
  catch(e){viewEl.innerHTML=`<div class="panel"><div class="lbl">Narration</div><p style="margin-top:8px">${esc(e.message)}</p>
    <button class="btn acc" style="margin-top:12px" onclick="location.hash='#/panels/${session}'">Review panels first</button></div>`;return}
  viewEl.innerHTML=`
  <div style="display:flex;align-items:end;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:14px">
    <div><div class="lbl">narration studio · ${esc(session)}</div>
      <h2 class="h-disp" style="font-size:28px;margin-top:4px">${NR.data.total} panels · ${NR.data.edited_count} edited</h2></div>
    <div style="display:flex;gap:8px;align-items:center">
      <label class="field" style="flex-direction:row;align-items:center;gap:8px"><span class="lbl">style</span>
        <select id="nrStyle">${["recap","literal"].map(s=>`<option ${s===NR.data.style?"selected":""}>${s}</option>`).join("")}</select></label>
      <button class="btn" id="nrRunAI" title="Run the Gemini narration stage for panels without narration">Generate narration (AI)</button>
      <button class="btn acc" id="nrToVoice">Narrator Studio →</button>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:250px minmax(0,.9fr) minmax(0,1.1fr);gap:var(--gut);align-items:start" id="nrGrid">
    <aside class="panel" style="padding:10px;max-height:70vh;overflow:auto" id="nrList"></aside>
    <section class="panel" style="padding:10px" id="nrImg"></section>
    <section class="panel" id="nrScript"></section>
  </div>`;
  $("#nrStyle").onchange=async e=>{try{NR.data=await api.narrStyle(NR.session,e.target.value);toast(`Style: ${e.target.value}`);drawNrList()}catch(err){toast(err.message,"err")}};
  $("#nrToVoice").onclick=()=>nav("voice",NR.session);
  $("#nrRunAI").onclick=async()=>{
    try{toast("Narration stage started");
      const s=store.settings;
      const{job_id}=await api.run({session:NR.session,order:null,tts:s.tts,voice:s.voice,style:NR.data.style,
        backend:s.backend,api_key:s.api_key,model:s.model,endpoint:s.endpoint,cf_account_id:s.cf_account_id,
        start_stage:"gemini_narration"});
      store.genJob[NR.session]=job_id;save();watch(job_id);nav("generate",NR.session);
    }catch(e){toast(e.message,"err")}};
  drawNrList();
}
function drawNrList(){
  const P=NR.data.panels;if(!NR.sel)NR.sel=P[0]?.id||null;
  $("#nrList").innerHTML=`<div class="lbl" style="padding:4px 6px 8px">panels</div>`+P.map(p=>`
    <div class="step ${p.id===NR.sel?"running":""}" data-pid="${esc(p.id)}" style="cursor:pointer;grid-template-columns:auto 1fr auto">
      <span class="mark" style="width:18px;height:18px;font-size:10px">${p.panel_index}</span>
      <span class="name" style="font-size:12px">${esc(p.id)}${p.edited?' <span class="chip ok" style="font-size:9px">edited</span>':""}</span>
      <span class="chip ${p.narration?"ok":"warn"}" style="font-size:9px">${p.narration?"✓":"empty"}</span>
    </div>`).join("");
  $$("#nrList .step").forEach(el=>el.onclick=()=>{NR.sel=el.dataset.pid;drawNrList()});
  drawNrPanel();
}
function drawNrPanel(){
  const p=NR.data.panels.find(x=>x.id===NR.sel);
  const img=$("#nrImg"),sc=$("#nrScript");
  if(!p){img.innerHTML=`<p style="color:var(--txt3)">no panels</p>`;sc.innerHTML="";return}
  img.innerHTML=`<div class="lbl" style="margin-bottom:8px">panel ${p.panel_index} · ${esc(p.id)}</div>
    <div style="background:#0D0F13;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;border:1px solid var(--line)">
      <img src="${fileURL(NR.session,p.image_file)}" style="max-width:100%;max-height:60vh;object-fit:contain" alt="Panel ${p.panel_index}"></div>
    ${p.dialogue?`<div style="margin-top:10px"><div class="lbl">dialogue</div><p style="font-size:12px;color:var(--txt2);margin-top:5px;max-height:9em;overflow:auto">${esc(p.dialogue)}</p></div>`:""}`;
  sc.innerHTML=`
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
      <div class="lbl">script</div><span class="lbl" id="nrSaveState"></span></div>
    <textarea id="nrTxt" style="width:100%;margin-top:8px;min-height:130px;font-family:var(--body);font-size:13.5px;color:var(--txt);background:var(--ink3);border:1px solid var(--line2);border-radius:var(--r);padding:9px 11px;resize:vertical;line-height:1.45">${esc(p.narration)}</textarea>
    <div class="auto" style="margin-top:6px">AI wrote: <b>${esc(p.ai_text)||"—"}</b></div>
    <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
      <button class="btn acc sm" id="nrSave">Save</button>
      <button class="btn sm" id="nrRegen" title="Rewrite with Gemini (needs API key)">↻ Regenerate</button>
      <button class="btn ghost sm" id="nrReset" ${p.edited?"":"disabled"}>↺ Restore AI text</button>
    </div>`;
  const save=async()=>{
    $("#nrSaveState").textContent="saving…";
    try{NR.data=await api.narrText(NR.session,p.id,$("#nrTxt").value);$("#nrSaveState").textContent="saved ✓";drawNrList()}
    catch(e){$("#nrSaveState").textContent="";toast(e.message,"err")}};
  $("#nrSave").onclick=save;
  $("#nrTxt").addEventListener("input",()=>{clearTimeout(NR.saveT);$("#nrSaveState").textContent="…";NR.saveT=setTimeout(save,900)});
  $("#nrRegen").onclick=async()=>{ $("#nrRegen").disabled=true;$("#nrSaveState").textContent="regenerating…";
    try{NR.data=await api.narrRegen(NR.session,p.id);$("#nrSaveState").textContent="regenerated ✓";drawNrList()}
    catch(e){toast(e.message,"err")}
    $("#nrRegen").disabled=false};
  $("#nrReset").onclick=async()=>{try{NR.data=await api.narrReset(NR.session,p.id);toast("Restored AI text","warn");drawNrList()}catch(e){toast(e.message,"err")}};
}
/* ================================================================
   VIEW — Narrator Studio (provider · voice · rate/pitch · preview)
   Real contract: /api/voice/{s} GET/PUT, /api/voices, /api/voice/{s}/preview
   ================================================================ */
let VS=null; /* {session,cfg,voices,previewUrl} */
const VOICE_PRESETS=[
  ["Natural",   0,  0,  1.0],["Fast",     15, 0,  1.0],
  ["Dramatic", -8, -10, 1.0],["Calm",    -8,  0,  1.0],
  ["Energetic",12, 15,  1.0],["Deep",    -5, -25, 1.0]];
async function vVoice(session){
  session=await requireSession(session);if(!session)return studioEmpty("Narrator Studio");
  setSession(session);
  viewEl.innerHTML=`
  <div class="lbl">narrator studio · saved per project in voice.json</div>
  <h2 class="h-disp" style="font-size:28px;margin:4px 0 16px">Voice</h2>
  <div class="vs-grid">
    <section class="panel lift">
      <div class="field"><span class="lbl">provider</span>
        <select id="vsProvider"><option value="edge">Edge TTS</option><option value="none">None (silent)</option></select></div>
      <div class="field" style="margin-top:12px"><span class="lbl">voice <span id="vsCount"></span></span>
        <input id="vsSearch" placeholder="search voices (e.g. Aria, en-US)…" style="margin-bottom:6px">
        <select id="vsVoice" size="9" style="min-height:190px"></select></div>
      <div class="field" style="margin-top:12px"><span class="lbl">rate <b id="vsRateV" style="color:var(--acc)">+0%</b></span>
        <input type="range" id="vsRate" min="-50" max="50" step="1" value="0"></div>
      <div class="field" style="margin-top:12px"><span class="lbl">pitch <b id="vsPitchV" style="color:var(--acc)">+0Hz</b></span>
        <input type="range" id="vsPitch" min="-50" max="50" step="1" value="0"></div>
      <div class="field" style="margin-top:12px"><span class="lbl">speed <b id="vsSpeedV" style="color:var(--acc)">1.0×</b></span>
        <input type="range" id="vsSpeed" min="0.5" max="1.6" step="0.05" value="1">
        <p class="auto">Edge TTS has one speech-rate control — speed is folded into rate on synth.</p></div>
      <div class="lbl" style="margin-top:16px">presets</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px" id="vsPresets">
        ${VOICE_PRESETS.map(p=>`<button class="btn sm" data-preset="${p[0]}">${p[0]}</button>`).join("")}
      </div>
    </section>
    <section class="panel lift" style="--d:80ms">
      <div class="lbl">preview text</div>
      <textarea id="vsText" style="margin-top:8px;min-height:70px"></textarea>
      <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
        <button class="btn acc" id="vsPreview">▶ Generate Preview</button>
        <button class="btn" id="vsApply">Use This Voice</button>
      </div>
      <div id="vsAudioWrap" style="margin-top:12px;display:none">
        <audio id="vsAudio" controls style="width:100%"></audio>
      </div>
      <p class="auto" id="vsState" style="margin-top:10px"></p>
      <hr class="cutline">
      <div class="lbl">voice comparison</div>
      <p style="font-size:12px;color:var(--txt3);margin:6px 0 10px">Preview the sample with other voices — only short clips are synthesized, cached by content.</p>
      <div id="vsCompare" style="display:grid;gap:6px"></div>
    </section>
  </div>`;
  VS={session,cfg:await api.voiceGet(session),voices:null,previewUrl:null};
  const $v=s=>$(s);
  const paint=()=>{
    $v("#vsProvider").value=VS.cfg.provider||"edge";
    $v("#vsRate").value=VS.cfg.rate??0; $v("#vsPitch").value=VS.cfg.pitch??0; $v("#vsSpeed").value=VS.cfg.speed??1;
    $v("#vsRateV").textContent=(VS.cfg.rate>=0?"+":"")+(VS.cfg.rate??0)+"%";
    $v("#vsPitchV").textContent=(VS.cfg.pitch>=0?"+":"")+(VS.cfg.pitch??0)+"Hz";
    $v("#vsSpeedV").textContent=(VS.cfg.speed??1).toFixed(2)+"×";
    if(!$v("#vsVoice").value)paintVoiceList();
  };
  const paintVoiceList=()=>{
    const sel=$v("#vsVoice"),q=($v("#vsSearch").value||"").toLowerCase();
    const all=VS.voices||[],cur=VS.cfg.voice||"en-US-AriaNeural";
    if(!all.length){sel.innerHTML=`<option selected>${esc(cur)}</option>`;
      $v("#vsCount").textContent="(catalogue unavailable — manual id)";return}
    const filtered=q?all.filter(v=>(v.shortname+" "+v.locale).toLowerCase().includes(q))
                    :all.filter(v=>v.language==="en");
    const groups={};filtered.forEach(v=>(groups[v.locale]=groups[v.locale]||[]).push(v));
    sel.innerHTML=Object.keys(groups).sort().map(loc=>
      `<optgroup label="${esc(loc)}">${groups[loc].map(v=>
        `<option value="${esc(v.shortname)}" ${v.shortname===cur?"selected":""}>${esc(v.shortname.replace("Neural",""))} · ${esc(v.gender[0]||"?")}</option>`).join("")}</optgroup>`).join("");
    $v("#vsCount").textContent=`(${filtered.length}${q?"":" English"} of ${all.length})`;
    VS.cfg.voice=sel.value||cur;
  };
  const preview=async cfg=>{
    $v("#vsState").textContent="synthesizing…";
    try{
      const r=await api.voicePreview(VS.session,cfg,$v("#vsText").value);
      $v("#vsAudio").src=r.url+"?t="+Date.now();$v("#vsAudioWrap").style.display="block";
      $v("#vsAudio").play().catch(()=>{});
      $v("#vsState").textContent=`preview ready · ${r.rate_cfg.rate} ${r.rate_cfg.pitch}`;
    }catch(e){$v("#vsState").textContent="";toast(e.message,"err")}
  };
  /* wiring */
  $v("#vsText").value="The protagonist suddenly realizes something is wrong. The city will never be the same again.";
  ["vsRate","vsPitch","vsSpeed"].forEach(id=>$v("#"+id).addEventListener("input",()=>{
    VS.cfg.rate=+$v("#vsRate").value;VS.cfg.pitch=+$v("#vsPitch").value;VS.cfg.speed=+$v("#vsSpeed").value;paint()}));
  $v("#vsSearch").addEventListener("input",paintVoiceList);
  $v("#vsProvider").addEventListener("change",()=>{
    VS.cfg.provider=$v("#vsProvider").value;
    $v("#vsPreview").disabled=$v("#vsApply").disabled=VS.cfg.provider==="none";
  });
  $v("#vsVoice").addEventListener("change",()=>{VS.cfg.voice=$v("#vsVoice").value});
  $v("#vsPresets").addEventListener("click",e=>{
    const p=VOICE_PRESETS.find(x=>x[0]===e.target.dataset.preset);if(!p)return;
    Object.assign(VS.cfg,{rate:p[1],pitch:p[2],speed:p[3]});paint();toast(`Preset: ${p[0]}`,"warn");
  });
  $v("#vsPreview").onclick=()=>preview(VS.cfg);
  $v("#vsApply").onclick=async()=>{
    try{VS.cfg=await api.voicePut(VS.session,VS.cfg);toast(`Voice saved: ${VS.cfg.voice}`)}
    catch(e){toast(e.message,"err")}};
  /* comparison: popular English voices, one ▶ per row — short cached clips only */
  const cmp=["en-US-AriaNeural","en-US-JennyNeural","en-US-GuyNeural","en-GB-RyanNeural"];
  $v("#vsCompare").innerHTML=cmp.map(v=>`
    <div style="display:flex;align-items:center;gap:8px">
      <button class="btn sm" data-cmp="${esc(v)}">▶</button>
      <span class="chip">${esc(v.replace("Neural",""))}</span>
    </div>`).join("");
  $v("#vsCompare").addEventListener("click",e=>{
    const v=e.target.dataset.cmp;if(!v)return;preview(Object.assign({},VS.cfg,{voice:v}))});
  paint();
  api.voiceList("edge").then(r=>{VS.voices=r.voices;paintVoiceList()}).catch(()=>paintVoiceList());
}
/* ================================================================
   VIEW — Generate (stage machine + run/cancel/retry)
   ================================================================ */
/* ================================================================
   VIEW — Generate (stage machine + run/cancel/retry)
   ================================================================ */
let genStop=null;
async function vGenerate(session){
  session=await requireSession(session);if(!session)return studioEmpty("Generate");
  setSession(session);
  viewEl.innerHTML=`
  <div class="gen">
    <section class="panel">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
        <div><div class="lbl">automation</div><h2 class="h-disp" style="font-size:26px;margin-top:2px">Run the machine</h2></div>
        <div style="display:flex;gap:8px" id="genBtns"></div>
      </div>
      <div class="progress" aria-hidden="true"><i id="pbar"></i></div>
      <div class="lbl" id="pPct" style="text-align:right">0%</div>
      <ol class="stepper" id="steps">${STAGES.map(([id,name])=>`
        <li class="step" data-stage="${id}"><span class="mark">·</span><span class="name">${name}</span><span class="chip" style="visibility:hidden">…</span></li>`).join("")}
      </ol>
      <div id="errSlot"></div>
    </section>
    <aside class="panel lift" style="position:sticky;top:78px">
      <div class="lbl">What happens</div>
      <p style="margin-top:10px;font-size:13px;color:var(--txt2)">Segment → Gemini narration → script → timeline → render → editor project. Cached stages are skipped on re-runs, so retries are cheap.</p>
      <hr class="cutline">
      <div class="lbl">Config in use</div>
      <div style="margin-top:10px;display:grid;gap:6px;font-family:var(--mono);font-size:11.5px;color:var(--txt2)">
        <div>backend&nbsp;: ${esc(store.settings.backend)}</div>
        <div>model&nbsp;&nbsp;&nbsp;: ${esc(store.settings.model||"(default)")}</div>
        <div>tts&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: ${esc(store.settings.tts)} · ${esc(store.settings.voice)}</div>
        <div>style&nbsp;&nbsp;&nbsp;: ${esc(store.settings.style)}</div>
        <div>order&nbsp;&nbsp;&nbsp;: ${store.order[session]?store.order[session].length+" panels (custom)":"auto"}</div>
      </div>
      <button class="btn ghost sm" style="margin-top:12px" onclick="location.hash='#/settings'">Change settings</button>
    </aside>
  </div>`;
  genStop&&genStop();
  paint(jobFor(session));
  watchActive(session);
}
function jobFor(session){                 /* latest generate job, else the segment job */
  const id=store.genJob[session];
  return id?api.job(id).catch(()=>api.job(session)):api.job(session);
}
async function watchActive(session){
  const paintLive=async()=>{try{paint(await jobFor(session))}catch(_){}};
  await paintLive();
  genStop=setInterval(async()=>{
    if(parseHash().view!=="generate"){clearInterval(genStop);return}
    await paintLive();
  },1200);
}
async function paint(jobP){
  const job=await Promise.resolve(jobP); if(!job||!$("#steps"))return;
  const idx=STAGES.findIndex(([id])=>id===job.stage);
  $$("#steps .step").forEach((el,i)=>{
    el.className="step "+(job.status==="completed"||i<idx||(i===idx&&job.status!=="running")?"done"
      :i===idx&&job.status==="failed"?"failed"
      :i===idx?"running":"");
    el.querySelector(".mark").textContent=job.status==="completed"||i<idx||(i===idx&&TERMINAL.includes(job.status)&&job.status!=="failed")?"✓"
      :(i===idx&&job.status==="failed")?"✕":(i===idx?"●":"·");
  });
  $("#pbar").style.width=(job.progress||0)+"%";
  $("#pPct").textContent=(job.progress||0)+"% · "+(job.stage?job.stage.replace(/_/g," "):job.status);
  $("#errSlot").innerHTML=job.status==="failed"&&job.error?`<div class="errbox">${esc(job.error)}</div>`:"";
  setPill(job.status, job.status==="running"?(job.stage||"running").replace(/_/g," "):job.status);
  /* context buttons */
  const running=job.status==="running", failed=job.status==="failed", done=job.status==="completed";
  $("#genBtns").innerHTML=
    running?`<button class="btn danger" id="bCancel">Cancel</button>`:
    failed ?`<button class="btn acc" id="bRun">Retry automation</button>`:
    done   ?`<button class="btn" id="bRerun">Re-run</button><button class="btn acc" id="bRev">Review →</button>`:
            `<button class="btn acc" id="bRun">Run Automation</button>`;
  const s=store.session;
  const run=async()=>{
    try{
      toast("Automation started");
      const{job_id}=await api.run({session:s,order:store.order[s]||null,
        tts:store.settings.tts,voice:store.settings.voice,style:store.settings.style,
        backend:store.settings.backend,api_key:store.settings.api_key,
        model:store.settings.model,endpoint:store.settings.endpoint,
        cf_account_id:store.settings.cf_account_id});
      store.genJob[s]=job_id;save();watch(job_id);paintLiveSoon();
    }catch(e){toast(e.message,"err")}
  };
  const paintLiveSoon=()=>setTimeout(()=>paint(api.job(store.genJob[s]||s)),400);
  $("#bRun")&&($("#bRun").onclick=run);
  $("#bRerun")&&($("#bRerun").onclick=run);
  $("#bCancel")&&($("#bCancel").onclick=async()=>{try{await api.cancel(store.genJob[s]||s);toast("Cancel requested","warn")}catch(e){toast(e.message,"err")}});
  $("#bRev")&&($("#bRev").onclick=()=>nav("review",s));
}
/* ================================================================
   VIEW — Review (video + synced narration cues)
   ================================================================ */
async function vReview(session){
  session=await requireSession(session);if(!session)return studioEmpty("Review");
  setSession(session);
  viewEl.innerHTML=`
  <div class="rev">
    <section>
      <div class="vwrap">
        <video id="vid" controls preload="metadata" src="${fileURL(session,"recap.mp4")}?t=${Date.now()}"></video>
        <div class="vmeta"><span>recap.mp4</span><span style="flex:1"></span>
          <a class="btn sm" download href="${fileURL(session,"recap.mp4")}">⤓ mp4</a>
          <a class="btn sm" download href="${fileURL(session,"recap.srt")}">⤓ srt</a>
          <button class="btn sm acc" id="toEd">Open in Editor</button>
        </div>
      </div>
    </section>
    <aside class="panel" style="align-self:start">
      <div class="lbl">Narration timeline</div>
      <div class="cues" id="cues" style="margin-top:12px"><p style="color:var(--txt3);font-size:13px">Loading cues…</p></div>
    </aside>
  </div>`;
  $("#toEd").onclick=()=>nav("editor",session);
  /* cues: timeline.json (times) joined with narration.json (text) */
  try{
    const[tl,nar]=await Promise.all([
      req(fileURL(session,"timeline.json")).catch(()=>null),
      req(fileURL(session,"narration.json")).catch(()=>null)]);
    const texts=Object.fromEntries((nar?.entries||[]).map(n=>[n.panel_id||n.id,n.text]));
    const cues=(tl?.entries||[]).map(e=>({s:e.start_seconds,e:e.start_seconds+e.duration_seconds,
      txt:texts[e.panel_id]||"",pid:e.panel_id})).filter(c=>c.txt);
    $("#cues").innerHTML=cues.length?cues.map((c,i)=>`
      <div class="cue" data-i="${i}"><time>${fmt(c.s)} → ${fmt(c.e)}</time>${esc(c.txt)}</div>`).join("")
      :`<p style="color:var(--txt3);font-size:13px">No cues found yet — run automation first, or wait for audio to land.</p>`;
    const vid=$("#vid");
    const click=el=>el.onclick=()=>{vid.currentTime=cues[+el.dataset.i].s+.01;vid.play()};
    $$(".cue").forEach(click);
    vid.addEventListener("timeupdate",()=>{
      const t=vid.currentTime;
      $$(".cue").forEach((el,i)=>el.classList.toggle("on",t>=cues[i].s&&t<cues[i].e));
    });
  }catch(_){$("#cues").innerHTML=`<p style="color:var(--coral);font-size:13px">No video yet — run automation first.</p>`}
}
/* ================================================================
   VIEW — Editor (preview / properties / timeline, one language)
   ================================================================ */
let ED=null; /* {session, project, sel:{type,id}} */
async function vEditor(session){
  session=await requireSession(session);if(!session)return studioEmpty("Editor");
  setSession(session);
  try{ED={session,project:await api.edGet(session).catch(async e=>{
      if(/not found/i.test(e.message))return api.edCreate(session);throw e}),sel:null};}
  catch(e){viewEl.innerHTML=`<div class="panel"><div class="lbl">Editor</div><p style="margin-top:8px">${esc(e.message)}</p>
    <button class="btn acc" style="margin-top:12px" onclick="location.hash='#/generate/${session}'">Run automation first</button></div>`;return}
  viewEl.innerHTML=`
  <div class="ed">
    <section class="preview">
      <video id="edVid" controls src="${fileURL(session,"recap_edited.mp4")}?t=${Date.now()}"
             onerror="this.src='${fileURL(session,"recap.mp4")}?t=${Date.now()}'"></video>
      <div class="transport">
        <button class="btn sm" id="tPlay" aria-label="Play/pause">▶</button>
        <input type="range" id="tSeek" min="0" max="1000" value="0" aria-label="Seek">
        <span class="tc" id="tTime">0:00.0 / 0:00.0</span>
      </div>
    </section>
    <aside class="props panel" id="props"></aside>
    <section class="tlwrap">
      <div class="tlhead">
        <span class="lbl">timeline · ${ED.project.edited_timeline.length} panels</span>
        <div style="display:flex;gap:6px;align-items:center">
          <span class="chip ${ED.project.needs_render?"warn":"ok"}" id="nrChip">${ED.project.needs_render?"needs render":"rendered"}</span>
          <button class="btn sm" id="bUndo" title="Ctrl+Z">↩ Undo</button>
          <button class="btn sm" id="bRedo" title="Ctrl+Shift+Z">↪ Redo</button>
          <button class="btn sm danger" id="bReset">Reset to Automated</button>
          <button class="btn sm acc" id="bRender">Render Video</button>
        </div>
      </div>
      <div class="tlscroll"><div id="tlInner"></div></div>
    </section>
  </div>`;
  drawEditor();
}
function tlTotal(){const t=ED.project.edited_timeline;const last=t[t.length-1];return last?last.start_seconds+last.duration_seconds:1}
function drawEditor(){
  const P=ED.project,total=tlTotal(),W=Math.max(900,total*90);
  const inner=$("#tlInner");inner.style.width=W+"px";
  const x=s=>(s/total*100)+"%";
  const startOf=pid=>{const e=P.edited_timeline.find(e=>e.panel_id===pid);return e?e.start_seconds:0};
  inner.innerHTML=`
    <div class="tlrow" id="rowPanels">
      ${P.edited_timeline.map(e=>`
        <div class="clip ${ED.sel?.type==="panel"&&ED.sel.id===e.panel_id?"sel":""}" data-pid="${esc(e.panel_id)}"
             style="left:${x(e.start_seconds)};width:${x(e.duration_seconds)}" tabindex="0"
             aria-label="Panel ${esc(e.panel_id)}, ${e.duration_seconds.toFixed(1)} seconds">
          <img loading="lazy" src="${fileURL(ED.session,(e.source_image||"").split("/").pop())}" alt="">
          <span class="cm">${esc(e.panel_id)} · ${e.duration_seconds.toFixed(1)}s</span>
          <span class="hd l" data-hd="l"></span><span class="hd r" data-hd="r"></span>
        </div>`).join("")}
      ${P.transitions.map((t,i)=>`<span class="trmark" data-tr="${i}" style="left:${x(startOf(t.to_panel_id))}" title="Transition"></span>`).join("")}
    </div>
    <div class="tlrow cap" id="rowCaps">
      ${P.captions.map(c=>`
        <div class="capchip ${ED.sel?.type==="caption"&&ED.sel.id===c.id?"sel":""}" data-cid="${esc(c.id)}"
             style="left:${x(c.start_seconds)};width:${x(Math.max(.4,c.end_seconds-c.start_seconds))}"
             title="${esc(c.text)}">${esc(c.text)}</div>`).join("")}
    </div>`;
  drawProps();
  /* transport */
  const vid=$("#edVid"),seek=$("#tSeek"),tc=$("#tTime");
  vid.ontimeupdate=()=>{seek.value=(vid.currentTime/Math.max(vid.duration||total,0.01))*1000;
    tc.textContent=`${fmt(vid.currentTime)} / ${fmt(vid.duration||total)}`};
  seek.oninput=()=>{vid.currentTime=(seek.value/1000)*(vid.duration||total)};
  $("#tPlay").onclick=()=>{vid.paused?vid.play():vid.pause();$("#tPlay").textContent=vid.paused?"▶":"❚❚"};
  /* selection + resize */
  $$(".clip",inner).forEach(c=>{
    c.onclick=e=>{if(e.target.dataset.hd)return;ED.sel={type:"panel",id:c.dataset.pid};drawEditor()};
    $$(".hd",c).forEach(h=>h.addEventListener("pointerdown",ev=>startResize(ev,c,h)));
  });
  $$(".capchip",inner).forEach(c=>c.onclick=()=>{ED.sel={type:"caption",id:c.dataset.cid};drawEditor()});
  $$(".trmark",inner).forEach(m=>m.onclick=()=>{ED.sel={type:"transition",id:+m.dataset.tr};drawEditor()});
  /* toolbar */
  $("#bUndo").disabled=P.history_index<0;
  $("#bRedo").disabled=P.history_index>=P.history.length-1;
  $("#bUndo").onclick=async()=>{await api.edPost(ED.session,"undo");refreshEd()};
  $("#bRedo").onclick=async()=>{await api.edPost(ED.session,"redo");refreshEd()};
  $("#bReset").onclick=async()=>{if(confirm("Reset ALL edits back to the automated result?")){await api.edPost(ED.session,"reset");refreshEd();toast("Reset to automated","warn")}};
  $("#bRender").onclick=async()=>{
    try{toast("Render started — this can take a while");
      const r=await api.edPost(ED.session,"render");
      if(r&&r.job_id)watch(r.job_id);
      setTimeout(async()=>{await refreshEd();$("#edVid").src=fileURL(ED.session,"recap_edited.mp4")+"?t="+Date.now();toast("Render finished")},1500);
    }catch(e){toast(e.message,"err")}
  };
}
function startResize(ev,clip,handle){
  ev.stopPropagation();
  const pid=clip.dataset.pid,total=tlTotal();
  const e=ED.project.edited_timeline.find(x=>x.panel_id===pid);
  const startX=ev.clientX, dur0=e.duration_seconds, pxPerSec=clip.offsetWidth/dur0;
  const move=m=>{
    const d=(m.clientX-startX)/pxPerSec*(handle.dataset.hd==="r"?1:-1);
    const nd=Math.max(0.5,dur0+d);
    e.duration_seconds=nd;clip.style.width=(nd/total*100)+"%";
    clip.querySelector(".cm").textContent=`${pid} · ${nd.toFixed(1)}s`;
  };
  const up=()=>{
    removeEventListener("pointermove",move);removeEventListener("pointerup",up);
    api.edPost(ED.session,"duration",{panel_id:pid,duration:Math.round(e.duration_seconds*100)/100})
      .then(refreshEd).catch(err=>toast(err.message,"err"));
  };
  addEventListener("pointermove",move);addEventListener("pointerup",up);
}
function drawProps(){
  const P=ED.project,el=$("#props");
  if(!ED.sel){el.innerHTML=`<div class="lbl">Properties</div>
    <p style="margin-top:10px;font-size:13px;color:var(--txt3)">Select a panel, caption or ◆ transition on the timeline.<br><br>
    <b style="color:var(--sky)">Blue values</b> are what the automation chose; your edits sit beside them, so you can always compare — and reset.</p>`;return}
  if(ED.sel.type==="panel"){
    const e=P.edited_timeline.find(x=>x.panel_id===ED.sel.id);
    const eff=P.effects.find(x=>x.panel_id===ED.sel.id)||{kind:(e.pan||{}).kind||"static"};
    el.innerHTML=`<div class="lbl">Panel ${esc(e.panel_id)}</div>
      <div class="grp"><div class="lbl">duration (s)</div>
        <input type="number" step="0.1" min="0.5" id="pDur" value="${e.duration_seconds.toFixed(2)}">
        <div class="auto">AI chose <b>${(e.automated_duration??e.duration_seconds).toFixed(2)}s</b></div></div>
      <div class="grp"><div class="lbl">effect / ken burns</div>
        <select id="pEff">${["static","pan_down","pan_right","pan_left","pan_up","zoom_in","zoom_out"]
          .map(k=>`<option ${k===eff.kind?"selected":""}>${k}</option>`).join("")}</select></div>
      <button class="btn acc sm" id="pApply">Apply</button>
      <button class="btn ghost sm" id="pAuto">↺ back to automated (${(e.automated_duration??e.duration_seconds).toFixed(1)}s)</button>`;
    $("#pApply").onclick=async()=>{
      await api.edPost(ED.session,"duration",{panel_id:e.panel_id,duration:parseFloat($("#pDur").value)||e.duration_seconds});
      await api.edPost(ED.session,"effect",{panel_id:e.panel_id,kind:$("#pEff").value,duration:parseFloat($("#pDur").value)||0});
      refreshEd();};
    $("#pAuto").onclick=async()=>{await api.edPost(ED.session,"duration",{panel_id:e.panel_id,duration:e.automated_duration??e.duration_seconds});refreshEd()};
  }
  if(ED.sel.type==="caption"){
    const c=P.captions.find(x=>x.id===ED.sel.id);
    el.innerHTML=`<div class="lbl">Caption ${esc(c.id)}</div>
      <div class="grp"><div class="lbl">text</div><textarea id="cTxt">${esc(c.text)}</textarea>
        <div class="auto">AI wrote <b>${esc(c.automated_text||"—")}</b></div></div>
      <div class="grp"><div class="lbl">start / end (s)</div>
        <div style="display:flex;gap:8px">
          <input type="number" step="0.1" id="cS" value="${c.start_seconds.toFixed(2)}">
          <input type="number" step="0.1" id="cE" value="${c.end_seconds.toFixed(2)}"></div></div>
      <button class="btn acc sm" id="cApply">Apply</button>
      <button class="btn ghost sm" id="cAuto">↺ back to automated text</button>`;
    $("#cApply").onclick=async()=>{await api.edPost(ED.session,"caption",
      {id:c.id,text:$("#cTxt").value,start_seconds:parseFloat($("#cS").value),end_seconds:parseFloat($("#cE").value)});refreshEd()};
    $("#cAuto").onclick=async()=>{await api.edPost(ED.session,"caption",{id:c.id,text:c.automated_text});refreshEd()};
  }
  if(ED.sel.type==="transition"){
    const t=P.transitions[ED.sel.id];
    el.innerHTML=`<div class="lbl">Transition ${esc(t.from_panel_id)} → ${esc(t.to_panel_id)}</div>
      <div class="grp"><div class="lbl">type</div>
        <select id="tType">${["cut","fade","crossfade"].map(k=>`<option ${k===t.type?"selected":""}>${k}</option>`).join("")}</select></div>
      <div class="grp"><div class="lbl">duration (s)</div>
        <input type="number" step="0.1" min="0" id="tDur" value="${(t.duration||0).toFixed(2)}"></div>
      <button class="btn acc sm" id="tApply">Apply</button>`;
    $("#tApply").onclick=async()=>{await api.edPost(ED.session,"transition",
      {from_panel_id:t.from_panel_id,to_panel_id:t.to_panel_id,type:$("#tType").value,duration:parseFloat($("#tDur").value)||0});refreshEd()};
  }
}
async function refreshEd(){ED.project=await api.edGet(ED.session);$("#nrChip").className="chip "+(ED.project.needs_render?"warn":"ok");
  $("#nrChip").textContent=ED.project.needs_render?"needs render":"rendered";drawEditor()}
addEventListener("keydown",e=>{
  if(parseHash().view!=="editor"||!ED)return;
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){e.preventDefault();e.shiftKey?$("#bRedo")?.click():$("#bUndo")?.click()}
});
/* ================================================================
   VIEW — Logs (structured, filterable, copy/download)
   ================================================================ */
let logTimer=null;
async function vLogs(){
  const jobs=store.recent.concat(Object.values(store.genJob||{}).map(id=>({session:id})))
    .filter((v,i,a)=>a.findIndex(x=>x.session===v.session)===i).slice(0,10);
  viewEl.innerHTML=`
  <div class="lbl">debug · structured job logs</div>
  <h2 class="h-disp" style="font-size:28px;margin:4px 0 16px">Logs</h2>
  <div class="logbar">
    <select id="logJob" style="padding:7px 10px;background:var(--ink3);color:var(--txt);border:1px solid var(--line2);border-radius:8px">
      ${jobs.map(j=>`<option value="${esc(j.session)}">${esc((store.names&&store.names[j.session])||j.session)}</option>`).join("")}
    </select>
    ${["All","INFO","WARNING","ERROR"].map((l,i)=>`<span class="chip ${i===0?"on":""}" data-l="${l}">${l}</span>`).join("")}
    <input id="logQ" placeholder="filter messages…" style="padding:7px 11px;background:var(--ink3);color:var(--txt);border:1px solid var(--line2);border-radius:8px">
    <button class="btn sm" id="logCopy">Copy</button>
    <button class="btn sm" id="logDl">Download</button>
  </div>
  <ol class="loglist" id="logList" aria-live="polite"></ol>`;
  let lvl="All",q="";
  const draw=async()=>{
    const id=$("#logJob").value;if(!id)return;
    try{
      const job=await api.jobLogs(id);
      const rows=(job.logs||[]).filter(r=>(lvl==="All"||r.level===lvl)&&(!q||(r.msg||"").toLowerCase().includes(q)));
      $("#logList").innerHTML=rows.map(r=>`
        <li class="${(r.level||"info").toLowerCase()}">
          <time>${new Date((r.t||0)*1000).toTimeString().slice(0,8)}</time>
          <span class="lv">${esc(r.level||"INFO")}</span>
          <span class="st">${esc(r.stage||"")}</span>
          <span class="msg">${esc(r.msg||"")}</span></li>`).join("")
        ||`<li><time></time><span class="lv"></span><span class="st"></span><span class="msg" style="color:var(--txt3)">no log lines</span></li>`;
      $("#logList").scrollTop=$("#logList").scrollHeight;
    }catch(e){$("#logList").innerHTML=`<li><span class="msg" style="color:var(--coral)">${esc(e.message)}</span></li>`}
  };
  $("#logJob").onchange=draw;
  $$(".logbar .chip").forEach(c=>c.onclick=()=>{$$(".logbar .chip").forEach(x=>x.classList.remove("on"));c.classList.add("on");lvl=c.dataset.l;draw()});
  $("#logQ").oninput=e=>{q=e.target.value.toLowerCase();draw()};
  $("#logCopy").onclick=async()=>{await navigator.clipboard.writeText($$("#logList .msg").map(m=>m.textContent).join("\n"));toast("Logs copied")};
  $("#logDl").onclick=()=>{
    const blob=new Blob([$$("#logList li").map(li=>li.textContent.trim()).join("\n")],{type:"text/plain"});
    const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`${$("#logJob").value}.log`;a.click();
  };
  draw();clearInterval(logTimer);logTimer=setInterval(()=>{if(parseHash().view==="logs")draw();else clearInterval(logTimer)},2000);
}
/* ================================================================
   VIEW — Settings
   ================================================================ */
async function vSettings(){
  const s=store.settings;
  viewEl.innerHTML=`
  <div class="lbl">settings · stored in this browser, sent only on /api/run</div>
  <h2 class="h-disp" style="font-size:28px;margin:4px 0 18px">Automation config</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;max-width:980px">
    <label class="field"><span class="lbl">backend</span>
      <select id="sBackend">${["gemini","openai","anthropic","ollama","cloudflare","none"].map(b=>`<option ${b===s.backend?"selected":""}>${b}</option>`).join("")}</select></label>
    <label class="field"><span class="lbl">model</span><input id="sModel" value="${esc(s.model)}" placeholder="e.g. gemini-2.5-flash"></label>
    <label class="field"><span class="lbl">api key (optional — .env works too)</span><input id="sKey" type="password" value="${esc(s.api_key)}" autocomplete="off"></label>
    <label class="field"><span class="lbl">endpoint / base url</span><input id="sEp" value="${esc(s.endpoint)}" placeholder="http://localhost:11434"></label>
    <label class="field"><span class="lbl">cloudflare account id</span><input id="sCf" value="${esc(s.cf_account_id)}"></label>
    <label class="field"><span class="lbl">tts</span><select id="sTts">${["edge","kokoro","none"].map(b=>`<option ${b===s.tts?"selected":""}>${b}</option>`).join("")}</select></label>
    <label class="field"><span class="lbl">voice</span><input id="sVoice" value="${esc(s.voice)}"></label>
    <label class="field"><span class="lbl">narration style</span><select id="sStyle">${["recap","literal"].map(b=>`<option ${b===s.style?"selected":""}>${b}</option>`).join("")}</select></label>
  </div>
  <div style="display:flex;gap:10px;margin-top:20px">
    <button class="btn acc" id="sSave">Save settings</button>
    <span class="gbadge" id="sG">checking gemini…</span>
  </div>
  <p style="margin-top:14px;font-size:12px;color:var(--txt3)">Keys are kept in localStorage and passed per-request; the server never persists them (see webapp/jobs.py — config is not serialised).</p>`;
  $("#sSave").onclick=()=>{
    Object.assign(store.settings,{backend:$("#sBackend").value,model:$("#sModel").value,api_key:$("#sKey").value,
      endpoint:$("#sEp").value,cf_account_id:$("#sCf").value,tts:$("#sTts").value,voice:$("#sVoice").value,style:$("#sStyle").value});
    save();toast("Settings saved");
  };
  api.config().then(c=>{const b=$("#sG");b.textContent=c.gemini_configured?`gemini ready · ${c.model}`:"gemini not configured";b.classList.toggle("ok",c.gemini_configured)});
}
/* ---------------- boot ---------------- */
if(store.session)$("#projBtn").textContent=store.names?.[store.session]||store.session;
$("#projBtn").onclick=()=>nav("studio");
render();
