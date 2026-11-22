"""DRAEM-inspired Perlin texture generation, adapted to the fixed historical head.

The NumPy gradient-noise construction follows VitjanZ/DRAEM perlin.py (MIT).
Only the anomaly generator is adapted. This is not a full DRAEM reproduction.
"""
import hashlib

import cv2 as cv
import numpy as np
import torch

from common import ROOT,OUT,read
from run_e3_cross_product import prepare_seed,array_hash

TEXTURES=ROOT/"bottle_cap_pilot/mouth_static_study/baseline_assets"


def perlin(rng,resolution):
    size=256
    delta=(resolution[0]/size,resolution[1]/size)
    repeat=(size//resolution[0],size//resolution[1])
    grid=np.mgrid[0:resolution[0]:delta[0],0:resolution[1]:delta[1]].transpose(1,2,0)%1
    angles=2*np.pi*rng.random((resolution[0]+1,resolution[1]+1))
    gradients=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    g00=gradients[:-1,:-1].repeat(repeat[0],0).repeat(repeat[1],1)
    g10=gradients[1:,:-1].repeat(repeat[0],0).repeat(repeat[1],1)
    g01=gradients[:-1,1:].repeat(repeat[0],0).repeat(repeat[1],1)
    g11=gradients[1:,1:].repeat(repeat[0],0).repeat(repeat[1],1)
    n00=(grid*g00).sum(-1)
    n10=(np.stack((grid[:,:,0]-1,grid[:,:,1]),-1)*g10).sum(-1)
    n01=(np.stack((grid[:,:,0],grid[:,:,1]-1),-1)*g01).sum(-1)
    n11=((grid-1)*g11).sum(-1)
    fade=6*grid**5-15*grid**4+10*grid**3
    low=(1-fade[:,:,0])*n00+fade[:,:,0]*n10
    high=(1-fade[:,:,0])*n01+fade[:,:,0]*n11
    return np.sqrt(2)*((1-fade[:,:,1])*low+fade[:,:,1]*high)


def replacement(product,group,operation,pool):
    number=int.from_bytes(hashlib.sha256(f"20261121:B:{product}:{group}:{operation}".encode()).digest()[:8],"little")
    rng=np.random.default_rng(number)
    item=pool[int(rng.integers(len(pool)))]
    path=TEXTURES/item["relative_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
    texture=cv.imdecode(np.fromfile(path,np.uint8),cv.IMREAD_COLOR)
    assert texture is not None
    texture=cv.resize(texture,(256,256),interpolation=cv.INTER_AREA)
    angle=float(rng.uniform(-45,45))
    texture=cv.warpAffine(texture,cv.getRotationMatrix2D((127.5,127.5),angle,1),(256,256),borderMode=cv.BORDER_REFLECT_101)
    gamma=float(rng.uniform(.5,2))
    texture=np.rint(255*(texture.astype(np.float32)/255)**gamma).astype(np.uint8)
    hsv=cv.cvtColor(texture,cv.COLOR_BGR2HSV).astype(np.int16)
    hue=int(rng.integers(-50,51))
    saturation=int(rng.integers(-50,51))
    hsv[:,:,0]=(hsv[:,:,0]+hue)%180
    hsv[:,:,1]=np.clip(hsv[:,:,1]+saturation,0,255)
    texture=cv.cvtColor(hsv.astype(np.uint8),cv.COLOR_HSV2BGR)
    fallback=False
    for attempt in range(8):
        resolution=2**rng.integers(0,6,size=2)
        noise=perlin(rng,resolution)
        rotation=float(rng.uniform(-90,90))
        noise=cv.warpAffine(noise,cv.getRotationMatrix2D((127.5,127.5),rotation,1),(256,256),borderMode=cv.BORDER_REFLECT_101)
        mask=(noise>.5).astype(np.uint8)
        if cv.resize(mask,(32,32),interpolation=cv.INTER_AREA).any():
            break
    else:
        yy,xx=np.mgrid[:256,:256]
        mask=((xx-128)**2+(yy-128)**2<=24**2).astype(np.uint8)
        fallback=True
    info=dict(texture_path=item["relative_path"],texture_sha256=item["sha256"],texture_angle=angle,
              gamma=gamma,hue=hue,saturation=saturation,mask_attempts=attempt+1,mask_fallback=fallback)
    return texture,mask,info


def prepare_alternate(images,manifest,seed):
    product=manifest[0]["product"]
    pool=read(OUT.parent/"TEXTURE_POOL.json")["files"]
    p=prepare_seed(images,manifest,seed,("S0","M"))
    p["reference_queries"]={arm:query.copy() for arm,query in p["queries"].items()}
    p["reference_audit"]=p["audit"]
    p["reference_targets"]=p["targets"].copy()
    counts=np.bincount(p["schedule"].numpy().ravel(),minlength=96)
    records=[]
    for local,index in enumerate(p["indices"]):
        source=images[index]
        group=manifest[index]["group_id"]
        for operation in range(3):
            q=local*8+5+operation
            texture,mask,info=replacement(product,group,operation,pool)
            strong=np.where(mask[:,:,None]>0,texture,source)
            weak=np.clip(np.rint((2/3)*source.astype(np.float32)+(1/3)*strong.astype(np.float32)),0,255).astype(np.uint8)
            p["targets"][q]=cv.resize(mask,(32,32),interpolation=cv.INTER_AREA).astype(np.float32).ravel()
            for arm,query,alpha in (("S0",strong,1.),("M",weak,1/3)):
                p["queries"][arm][q]=query
                delta=query.astype(float)-source
                ideal=alpha*(strong.astype(float)-source)
                assert np.all(delta[mask==0]==0) and np.abs(delta-ideal).max()<=.501
                records.append(dict(seed=seed,arm=arm,query=q,group=group,operation=f"texture{operation}",
                    draws=int(counts[q]),alpha=alpha,source_sha=array_hash(source),query_sha=array_hash(query),
                    mask_sha=array_hash(mask),mask_pixels=int(mask.sum()),mask_fraction=float(mask.mean()),
                    changed_pixels=int(np.any(delta!=0,axis=2).sum()),mae=float(np.abs(delta[mask>0]).mean()),
                    rounding=float(np.abs(delta-ideal).max()),**info))
    p["audit"]=records
    return p


def feature_audit(p,arm,desc,reference,std):
    rows=[]
    for q in range(96):
        if q%8<5:
            continue
        for name,feat,audit in (("alternate",desc,p["audit"]),("original",reference,p["reference_audit"])):
            record=next(r for r in audit if r["arm"]==arm and r["query"]==q)
            delta=feat[q]-feat[q//8*8]
            target=p["targets"][q] if name=="alternate" else p["reference_targets"][q]
            rows.append(dict(generator=name,arm=arm,query=q,group=record["group"],
                input_MAE=record["mae"],mask_grid_fraction=float(target.mean()),
                raw_feature_RMS=float(delta.square().mean().sqrt()),
                normalized_feature_RMS=float((delta/std).square().mean().sqrt())))
    return rows

