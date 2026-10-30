"""Exact rectangle geometry shared by both evaluation tasks."""
import math

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def intersect(a,b):return [max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
def enclosure(boxes):return [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
def inside(a,b):return area(intersect(a,b))/max(area(a),1e-9)
def iou(a,b):
    i=area(intersect(a,b));return i/max(area(a)+area(b)-i,1e-9)
def union_area(boxes):
    """Exact rectangle-union area; overlapping members never double counted."""
    boxes=[b for b in boxes if area(b)>0]
    xs=sorted({b[k] for b in boxes for k in (0,2)});total=0.
    for x1,x2 in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[3]) for b in boxes if b[0]<x2 and b[2]>x1)
        height=0.;end=-math.inf
        for y1,y2 in intervals:
            height+=max(0,y2-max(y1,end));end=max(end,y2)
        total+=(x2-x1)*height
    return total

