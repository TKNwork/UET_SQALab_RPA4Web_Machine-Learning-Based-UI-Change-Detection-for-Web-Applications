// Validate the GT visualization's interactions without browser access.
import fs from 'node:fs';
import vm from 'node:vm';
const html=fs.readFileSync('geometry/report_viewer/public/gt/index.html','utf8');
const raw=html.match(/<script type="application\/json" data-role="dataset">([\s\S]*?)<\/script>/)[1];
const dataset=JSON.parse(raw);
const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Node {
 constructor(tag='div'){this.tag=tag;this.children=[];this.attrs={};this.listeners={};this.classList={add:()=>{}};this.textContent='';}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,v){this.listeners[k]=v;}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=nodes;}
}
const roles={};
for(const role of ['dataset','totals','prev','pair','next','type','zoom','boxes','selection','before','after','gtlist','detail','overview'])roles[role]=new Node();
roles.dataset.textContent=raw;
const root={querySelector:s=>roles[s.match(/"(.*?)"/)[1]]};
const document={getElementById:()=>root,createElement:t=>new Node(t),createElementNS:(_,t)=>new Node(t),createTextNode:s=>({textContent:s})};
vm.runInNewContext(code,{document,window:{addEventListener:()=>{}},console});
if(roles.overview.children.length!==131)throw Error('Missing overview samples');
let total=0;
for(const sample of dataset.samples){
 roles.pair.listeners.change({target:{value:sample.id}});
 if(roles.gtlist.children.length!==sample.bboxes.length)throw Error('Missing GT list');
 for(const button of [...roles.gtlist.children]){
  button.listeners.click();roles.zoom.listeners.change({target:{value:'roi'}});
  for(const side of ['before','after']){
   const svg=roles[side].children[0];
   if(!svg.attrs.viewBox || svg.children.filter(n=>n.tag==='rect').length!==sample.bboxes.length)throw Error('Missing bbox or ROI');
  }
  total++;
 }
}
if(total!==297)throw Error('Missing GT');
for(const type of ['geometry','style','content','visibility']){
 roles.type.listeners.change({target:{value:type}});
 const expected=dataset.samples.filter(s=>s.bboxes.some(g=>g.type.includes(type))).length;
 if(roles.overview.children.length!==expected)throw Error('Type filter mismatch');
}
roles.type.listeners.change({target:{value:'all'}});
roles.boxes.listeners.change({target:{checked:false}});
if(roles.before.children[0].children.some(n=>n.tag==='rect'))throw Error('Hide bbox failed');
console.log('PASS: all 131 samples, 297 GT selections, before/after ROI, 4 type filters, hide boxes.');
