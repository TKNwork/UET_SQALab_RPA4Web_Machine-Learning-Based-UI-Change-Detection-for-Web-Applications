// Exercise all static content-viewer menus and row-to-evidence/overlay links.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
const root=process.argv[2]||'content/results/organized_v1';
const manifest=JSON.parse(fs.readFileSync(path.join(root,'data_manifest.json'),'utf8'));
class Node {
  constructor(tag='div'){this.tag=tag;this.children=[];this.attrs={};this.textContent='';this.checked=false;this.value='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=[...nodes];}
  setAttribute(key,value){this.attrs[key]=String(value);}
}
let samples=0,totalRows=0;
for(const sample of manifest.samples){
  const file=path.join(root,sample.pair_id,'review.html');if(!fs.existsSync(file))continue;
  const html=fs.readFileSync(file,'utf8');
  const code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const nodes=Object.fromEntries(['gt','det','filter','before','after','rows','detail'].map(k=>[k,new Node(k)]));
  nodes.gt.checked=true;nodes.filter.value='predictions';
  const document={getElementById:id=>nodes[id],createElement:tag=>new Node(tag),createElementNS:(_,tag)=>new Node(tag)};
  const context=vm.createContext({document,console});vm.runInContext(code,context);
  const payload=vm.runInContext('data',context);
  for(const menu of ['predictions','matched_pairs','uncertain_pairs','unmatched_before','unmatched_after','gt_coverage']){
    nodes.filter.value=menu;nodes.filter.onchange();
    const items=menu==='gt_coverage'?payload.evaluation[menu]:payload.inference[menu];
    if(nodes.rows.children.length!==items.length)throw Error('Wrong rows '+sample.pair_id+' '+menu);
    for(let i=0;i<items.length;i++){
      nodes.rows.children[i].onclick();
      if(JSON.stringify(JSON.parse(nodes.detail.textContent))!==JSON.stringify(items[i]))throw Error('Wrong detail');
      if(nodes.before.children[0].attrs.href!=='before.png'||nodes.after.children[0].attrs.href!=='after.png')throw Error('Wrong images');
      for(const side of ['before','after']){
        const count=nodes[side].children.filter(n=>n.tag==='rect').length;
        if(count<payload.evaluation.gt_count)throw Error('Missing GT boxes');
      }
      totalRows++;
    }
  }
  nodes.gt.checked=false;nodes.det.checked=true;nodes.det.onchange();
  for(const side of ['before','after']){
    const count=nodes[side].children.filter(n=>n.tag==='rect').length;
    if(count<payload.inference[side+'_detections'].length)throw Error('Missing detectors');
    if(!fs.existsSync(path.join(root,sample.pair_id,side+'.png')))throw Error('Missing image file');
  }
  samples++;
}
console.log(JSON.stringify({samples,selected_rows:totalRows,menus_per_sample:6,gt_and_detection_toggle:true,passed:true}));
