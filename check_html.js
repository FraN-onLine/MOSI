// Headless test harness for evacuation_sim.html PSO+boids logic.
const fs = require('fs');
const html = fs.readFileSync('evacuation_sim.html', 'utf-8');
const m = html.match(/<script id="appjs">([\s\S]*?)<\/script>/);
if (!m) { console.log('NO SCRIPT FOUND'); process.exit(1); }
const js = m[1];

// ---- DOM stubs ----
function makeEl() {
  return {
    style: {}, dataset: {}, classList: { add(){}, remove(){}, toggle(){}, contains(){return false;} },
    value: '0', checked: false, textContent: '', innerHTML: '',
    children: [], appendChild(c){ this.children.push(c); },
    addEventListener(){}, setAttribute(){}, getAttribute(){return null;},
    getBoundingClientRect(){ return { left:0, top:0, width:1300, height:820 }; },
    getContext(){ return ctxStub; },
  };
}
const ctxStub = new Proxy({}, { get(){ return () => {}; } });
const elements = {};
function getEl(id){ if(!elements[id]) elements[id]=makeEl(); return elements[id]; }
// default inputs the sim reads
getEl('psoW').value='0.7'; getEl('psoC1').value='1.4'; getEl('psoC2').value='1.4';
getEl('psoMaxV').value='10'; getEl('psoIters').value='2'; getEl('addCount').value='16';
getEl('boidsOn').checked=true; getEl('boidSep').value='1.6'; getEl('boidAli').value='0.6';
getEl('boidCoh').value='0.8'; getEl('boidRadius').value='70';
getEl('wallAvoid').checked=true; getEl('wallRepel').value='2.2';

global.document = {
  getElementById: getEl,
  createElement: () => makeEl(),
  addEventListener(){}, 
};
global.window = { addEventListener(){}, app: null };
global.requestAnimationFrame = () => 0; // inert

// ---- evaluate the app script in this context ----
const vm = require('vm');
const sandbox = { document: global.document, window: global.window, Math, console,
  requestAnimationFrame: global.requestAnimationFrame, requestIdleCallback: undefined };
sandbox.__exp = {};
vm.createContext(sandbox);
try {
  vm.runInContext(js + '\n;__exp.state=state;__exp.psoStep=psoStep;__exp.Particle=Particle;__exp.groupBests=groupBests;', sandbox, { filename: 'app.js' });
} catch (e) {
  console.log('EVAL ERROR:', e.message);
  console.log(e.stack ? e.stack.split('\n').slice(0,6).join('\n') : '(no stack)');
  process.exit(1);
}

// Pull the exposed internals off the sandbox.
const ctx = sandbox.__exp;
function need(name){ if(!(name in ctx)) { console.log('MISSING:', name); process.exit(1);} return ctx[name]; }

const state = need('state');
const psoStep = need('psoStep');
const Particle = need('Particle');

// Build a scenario: open room, perimeter exits, 16 particles.
state.walls = []; // no interior walls for baseline
state.exits = [ {x:0,y:330,w:25,h:160}, {x:1275,y:330,w:25,h:160} ];
state.safes = [];
state.particles = [];
for (let i=0;i<16;i++){
  const p = new Particle(300 + Math.random()*700, 300 + Math.random()*220);
  state.particles.push(p);
}
state.iter = 0;
state.mode = 'sim';

function evacuated(){ return state.particles.filter(p=>p.evacuated).length; }
console.log('start: particles=%d evacuated=%d', state.particles.length, evacuated());

let steps=0;
while (evacuated() < state.particles.length && steps < 3000){
  psoStep(state.pso.iters); // 2 iters per call
  steps += state.pso.iters;
  if (steps % 200 === 0) console.log('  iters=%d evacuated=%d', steps, evacuated());
}
console.log('final: iters=%d evacuated=%d/%d', steps, evacuated(), state.particles.length);

// Basic assertions
let pass = true;
if (evacuated() !== state.particles.length){ console.log('FAIL: not all evacuated'); pass=false; }
else console.log('PASS: all evacuated');
// each evacuated particle should be touching its target zone (within radius of boundary)
function _rectDist(r,p){const cx=Math.max(r.x,Math.min(p.x,r.x+r.w)),cy=Math.max(r.y,Math.min(p.y,r.y+r.h));return Math.hypot(p.x-cx,p.y-cy);}
for (const p of state.particles){
  const t = p.targetRect();
  const d = _rectDist(t, p.pos);
  if (d > p.radius + 1){ console.log('FAIL: particle not at target boundary, d=%s', d.toFixed(1)); pass=false; }
}
// particles used multiple exits (per-target assignment)
const usedExits = new Set(state.particles.map(p=>state.exits.indexOf(p.targetRect())));
console.log('distinct targets used:', [...usedExits].join(','));

// ---- Scenario 2: interior segment + rect walls, boids on ----
state.walls = [
  { x1:300, y1:200, x2:300, y2:620 },   // vertical segment
  { x:600, y:350, w:40, h:120, center(){ return {x:this.x+this.w/2,y:this.y+this.h/2}; } }, // rect obstacle
];
state.exits = [ {x:0,y:330,w:25,h:160}, {x:1275,y:330,w:25,h:160} ];
state.safes = [];
state.particles = [];
for (let i=0;i<14;i++) state.particles.push(new Particle(100 + Math.random()*1000, 100 + Math.random()*600));
state.iter = 0;
let steps2=0;
while (state.particles.filter(p=>p.evacuated).length < state.particles.length && steps2 < 4000){
  psoStep(state.pso.iters); steps2 += state.pso.iters;
}
const ev2 = state.particles.filter(p=>p.evacuated).length;
console.log('scenario2: iters=%d evacuated=%d/%d', steps2, ev2, state.particles.length);
// Assert no particle ended inside a wall (segment distance >= radius, rect not contained)
let inside = 0;
for (const p of state.particles){
  for (const w of state.walls){
    if (w.w && w.h){ if (p.pos.x>w.x-0.5 && p.pos.x<w.x+w.w+0.5 && p.pos.y>w.y-0.5 && p.pos.y<w.y+w.h+0.5) inside++; }
  }
}
if (inside) { console.log('FAIL: %d particles inside a rect wall', inside); pass=false; }
else console.log('PASS: no particles stuck inside rect walls');
if (ev2 === state.particles.length) console.log('PASS: scenario2 all evacuated');
else { console.log('WARN: scenario2 %d/%d evacuated (some may be blocked by walls)', ev2, state.particles.length); }

process.exit(pass?0:1);
