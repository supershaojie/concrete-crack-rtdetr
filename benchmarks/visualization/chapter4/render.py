"""Three-rule, aspect-preserving matrices; vector labels/boxes in PDF and SVG."""
from __future__ import annotations
import base64
from html import escape
import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import NAMES, read_json, sha256, write_json
from .geometry import inverse_cam, normalize_cam

GT_COLOR = "#00A6A6"
PRED_COLOR = "#D62728"


def font_paths():
    pairs = [("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"),
             ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
             ("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
             ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")]
    base_fonts=Path(sys.base_prefix)/f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/matplotlib/mpl-data/fonts/ttf"
    pairs.append((str(base_fonts/"DejaVuSerif.ttf"),str(base_fonts/"DejaVuSerif-Bold.ttf")))
    for regular, bold in pairs:
        if Path(regular).is_file() and Path(bold).is_file():
            return regular, bold
    raise FileNotFoundError("Install fonts-liberation or Times New Roman; no silent sans-serif substitution")


def jet(values):
    """Frozen 256-entry JET LUT, linear-segment definition (same LUT for every panel)."""
    positions = np.linspace(0, 1, 256)
    control = [[(0,0),(.35,0),(.66,1),(.89,1),(1,.5)],
               [(0,0),(.125,0),(.375,1),(.64,1),(.91,0),(1,0)],
               [(0,.5),(.11,1),(.34,1),(.65,0),(1,0)]]
    lut = np.stack([np.interp(positions, *np.array(channel).T) for channel in control], axis=1)
    return lut[np.round(np.clip(values, 0, 1)*255).astype(np.uint8)]


def cam_overlay(image, archive, metadata, alpha):
    with np.load(archive, allow_pickle=False) as arrays:
        native = [(k,arrays[k]) for k in sorted(arrays.files) if k.startswith("cam_")]
    if not native:
        raise ValueError("Missing native CAM arrays")
    maps = []
    layer_status={f"cam_s{row['stride']}":row["status"] for row in metadata.get("layers",[])}
    for name,value in native:
        # Normalize each scale BEFORE fixed equal-weight aggregation, including zero unused scales.
        if value.max() <= 1e-12 or layer_status.get(name,"passed")!="passed":
            maps.append(np.zeros((image.height, image.width), dtype=np.float32))
        else:
            value, _ = normalize_cam(value)
            maps.append(inverse_cam(value, metadata["geometry"]))
    full, normalization = normalize_cam(np.mean(maps, axis=0))
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)/255
    return Image.fromarray(np.round(np.clip((1-alpha)*rgb+alpha*jet(full), 0, 1)*255).astype(np.uint8)), normalization


class Canvas:
    def __init__(self, width, height, dpi):
        self.width, self.height, self.dpi = width, height, dpi
        self.ops = []
        self.regular, self.bold = font_paths()

    def text(self, x, y, text, size=8.5, bold=False, anchor="start", color="#000000"):
        self.ops.append(("text", x,y,str(text),size,bold,anchor,color))

    def line(self, x1, y1, x2, y2, color="#000000", width=.55):
        self.ops.append(("line", x1,y1,x2,y2,color,width))

    def rect(self, x, y, w, h, color, width=.65):
        self.ops.append(("rect",x,y,w,h,color,width))

    def image(self, image, x, y, w, h):
        # Only retain the pixels the final raster can resolve, not full-size duplicate originals.
        image = image.copy()
        image.thumbnail((max(1, round(w*self.dpi/72)), max(1, round(h*self.dpi/72))), Image.Resampling.LANCZOS)
        data = io.BytesIO()
        image.convert("RGB").save(data, format="JPEG", quality=93, subsampling=0)
        self.ops.append(("image",x,y,w,h,data.getvalue()))

    def export(self, stem, formats=("png","pdf","svg")):
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        scale = self.dpi/72
        regular, bold = self.regular, self.bold
        if "png" in formats:
            raster = Image.new("RGB", (round(self.width*scale),round(self.height*scale)), "white")
            draw = ImageDraw.Draw(raster)
            for op in self.ops:
                if op[0] == "image":
                    _,x,y,w,h,data=op
                    with Image.open(io.BytesIO(data)) as im:
                        im=im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
                        raster.paste(im,(round(x*scale),round(y*scale)))
                elif op[0] == "text":
                    _,x,y,text,size,is_bold,anchor,color=op
                    draw.text((x*scale,y*scale),text,font=ImageFont.truetype(bold if is_bold else regular,round(size*scale)),
                              fill=color,anchor={"start":"ls","middle":"ms","end":"rs"}[anchor])
                elif op[0] == "line":
                    _,x1,y1,x2,y2,color,width=op
                    draw.line((x1*scale,y1*scale,x2*scale,y2*scale),fill=color,width=max(1,round(width*scale)))
                else:
                    _,x,y,w,h,color,width=op
                    draw.rectangle((x*scale,y*scale,(x+w)*scale,(y+h)*scale),outline=color,width=max(1,round(width*scale)))
            raster.save(stem.with_suffix(".png"), dpi=(self.dpi,self.dpi), optimize=True)
        if "svg" in formats:
            import hashlib
            family = "Times New Roman" if "times" in regular.lower() else "Liberation Serif" if "liberation" in regular.lower() else "DejaVu Serif"
            svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width/72}in" height="{self.height/72}in" viewBox="0 0 {self.width} {self.height}">',
                 '<rect width="100%" height="100%" fill="white"/>']
            definitions={};references={}
            for op in self.ops:
                if op[0]=="image":
                    _,x,y,w,h,data=op
                    key=(hashlib.sha256(data).hexdigest(),w,h)
                    if key not in definitions:
                        name=f"embedded_image_{len(definitions)}";definitions[key]=name
                        references[name]=f'<image id="{name}" width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode(data).decode()}"/>'
            svg.append("<defs>"+"".join(references.values())+"</defs>")
            for op in self.ops:
                if op[0]=="image":
                    _,x,y,w,h,data=op
                    name=definitions[(hashlib.sha256(data).hexdigest(),w,h)]
                    svg.append(f'<use href="#{name}" x="{x}" y="{y}"/>')
                elif op[0]=="text":
                    _,x,y,text,size,is_bold,anchor,color=op
                    svg.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{"bold" if is_bold else "normal"}" text-anchor="{anchor}" fill="{color}">{escape(text)}</text>')
                elif op[0]=="line":
                    _,x1,y1,x2,y2,color,width=op
                    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
                else:
                    _,x,y,w,h,color,width=op
                    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="{width}"/>')
            stem.with_suffix(".svg").write_text("\n".join(svg+["</svg>"]), encoding="utf-8")
        if "pdf" in formats:
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            pdfmetrics.registerFont(TTFont("Ch4Serif",regular))
            pdfmetrics.registerFont(TTFont("Ch4SerifBold",bold))
            pdf=canvas.Canvas(str(stem.with_suffix(".pdf")),pagesize=(self.width,self.height),pageCompression=1)
            for op in self.ops:
                if op[0]=="image":
                    _,x,y,w,h,data=op
                    pdf.drawImage(ImageReader(io.BytesIO(data)),x,self.height-y-h,w,h)
                elif op[0]=="text":
                    _,x,y,text,size,is_bold,anchor,color=op
                    pdf.setFont("Ch4SerifBold" if is_bold else "Ch4Serif",size)
                    pdf.setFillColor(color)
                    {"start":pdf.drawString,"middle":pdf.drawCentredString,"end":pdf.drawRightString}[anchor](x,self.height-y,text)
                elif op[0]=="line":
                    _,x1,y1,x2,y2,color,width=op
                    pdf.setStrokeColor(color);pdf.setLineWidth(width)
                    pdf.line(x1,self.height-y1,x2,self.height-y2)
                else:
                    _,x,y,w,h,color,width=op
                    pdf.setStrokeColor(color);pdf.setLineWidth(width)
                    pdf.rect(x,self.height-y-h,w,h,stroke=1,fill=0)
            pdf.save()
        record = dict(pixel_width=round(self.width*scale), pixel_height=round(self.height*scale), dpi=self.dpi,
                      width_inches=self.width/72,height_inches=self.height/72,font=regular,font_sha256=sha256(regular),
                      vectors="PDF/SVG text, rules, boxes and score labels; embedded image pixels")
        write_json(stem.with_suffix(".dimensions.json"),record)
        return record


def fitted(region, x, y, w, h):
    scale = min(w/(region[2]-region[0]),h/(region[3]-region[1]))
    iw,ih=(region[2]-region[0])*scale,(region[3]-region[1])*scale
    return x+(w-iw)/2,y+(h-ih)/2,iw,ih,scale


def boxes_on_canvas(canvas, boxes, region, placement, color, scores=False):
    x,y,w,h,scale=placement
    occupied=[]
    for row in boxes:
        x1,y1,x2,y2=row["bbox"]
        # Rendering clips graphics at the view boundary only; the source boxes stay untouched.
        x1,x2=max(region[0],x1),min(region[2],x2)
        y1,y2=max(region[1],y1),min(region[3],y2)
        if x2<=x1 or y2<=y1:
            continue
        bx,by=x+(x1-region[0])*scale,y+(y1-region[1])*scale
        bw,bh=(x2-x1)*scale,(y2-y1)*scale
        canvas.rect(bx,by,bw,bh,color)
        if scores:
            label=f"{row['score']:.2f}"
            lx=min(x+w-12,max(x,bx));ly=min(y+h-1,max(y+5,by+5))
            # Move labels into free vertical slots; never drop duplicate/FP detections.
            for _ in range(max(1,len(boxes))):
                if not any(abs(lx-a)<13 and abs(ly-b)<4.5 for a,b in occupied):
                    break
                ly=ly+5 if ly+5<y+h else y+5
            occupied.append((lx,ly))
            canvas.text(lx,ly,label,size=4.8,color=color)


def save_full_detection(original,boxes,path,scores):
    """Full original field of view for auditing, with the same threshold/color rules."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    panel=original.copy();draw=ImageDraw.Draw(panel)
    line=max(1,round(min(panel.size)/550));font_size=max(12,round(min(panel.size)/75))
    font=ImageFont.truetype(font_paths()[0],font_size)
    occupied=[]
    for row in boxes:
        x1,y1,x2,y2=row["bbox"]
        draw.rectangle((x1,y1,x2,y2),outline=PRED_COLOR if scores else GT_COLOR,width=line)
        if scores:
            x=min(max(0,x1),panel.width-font_size*2)
            y=min(max(0,y1),panel.height-font_size)
            for _ in range(max(1,len(boxes))):
                if not any(abs(x-a)<font_size*2 and abs(y-b)<font_size for a,b in occupied):break
                y=y+font_size if y+2*font_size<panel.height else 0
            occupied.append((x,y))
            draw.text((x,y),f"{row['score']:.2f}",fill=PRED_COLOR,font=font)
    panel.save(path,quality=90,subsampling=0,optimize=True)


def matrix(cases_doc, images, predictions, models, image_root, output, cam_index=None,
           heatmap=False, config=None, draft=False, formats=("png","pdf","svg")):
    config=config or {}
    cases=cases_doc["cases"]
    width=config.get("width_inches",7.2)*72
    max_height=config.get("max_height_inches",9.5)*72
    label=78.; margin=8.; gap=2.; header=23.;footer=24. if heatmap else 14.
    cell_w=(width-2*margin-label-gap*(len(cases)-1))/len(cases)
    cell_h=min(max(cell_w*(c["view_region"][3]-c["view_region"][1])/(c["view_region"][2]-c["view_region"][0]) for c in cases),
               (max_height-header-footer-2*margin-gap*len(models))/(len(models)+1))
    height=2*margin+header+footer+(len(models)+1)*cell_h+gap*len(models)
    canvas=Canvas(width,height,config.get("dpi",450))
    canvas.line(margin,margin,width-margin,margin,width=1.)
    canvas.line(margin,margin+header-3,width-margin,margin+header-3,width=.5)
    bottom=margin+header+(len(models)+1)*cell_h+gap*len(models)
    canvas.line(margin,bottom+3,width-margin,bottom+3,width=1.)
    labels=["Original" if heatmap else "Ground truth"]+[m["model"] for m in models]
    pending=[];normalizations=[]
    for j,case in enumerate(cases):
        key=case["image"];region=case["view_region"]
        canvas.text(margin+label+j*(cell_w+gap)+cell_w/2,margin+14,f"Image {j+1}",anchor="middle")
        with Image.open(Path(image_root)/Path(key).name) as im:
            original=im.convert("RGB")
        for i,row_label in enumerate(labels):
            x=margin+label+j*(cell_w+gap);y=margin+header+i*(cell_h+gap)
            if j==0:
                canvas.text(margin+label-6,y+cell_h/2+3,row_label,bold=row_label=="ours",anchor="end")
            placement=fitted(region,x,y,cell_w,cell_h)
            panel=original
            if not heatmap and i>0 and key not in predictions.get(models[i-1]["id"],{}):
                if not draft:raise ValueError("Final detection panel lacks verified predictions")
                pending.append(dict(model_id=models[i-1]["id"],image=key,status="prediction_pending"))
                panel=Image.new("RGB",original.size,"#EEEEEE")
                canvas.image(panel.crop(region),*placement[:4])
                canvas.text(x+cell_w/2,y+cell_h/2,"PREDICTION PENDING",size=6,anchor="middle",color="#9C2020")
                continue
            if heatmap and i>0:
                mid=models[i-1]["id"]
                entry=(cam_index or {}).get(mid,{}).get(key)
                if not entry or entry.get("status")!="passed":
                    if not draft:
                        raise ValueError("Final heatmap missing/invalid: "+mid+" "+key)
                    pending.append(dict(model_id=mid,image=key,status=(entry or {}).get("status","not_computed")))
                    # Explicit neutral placeholder: never display the original as a successful heatmap.
                    panel=Image.new("RGB",original.size,"#EEEEEE")
                    canvas.image(panel.crop(region),*placement[:4])
                    canvas.text(x+cell_w/2,y+cell_h/2,"CAM PENDING / FAILED",size=6,anchor="middle",color="#9C2020")
                    continue
                try:
                    panel,normalization=cam_overlay(original,entry["npz"],read_json(entry["metadata"]),config.get("overlay_alpha",.45))
                except Exception as error:
                    if not draft:raise
                    pending.append(dict(model_id=mid,image=key,status="render_failed",error=repr(error)))
                    panel=Image.new("RGB",original.size,"#EEEEEE")
                    canvas.image(panel.crop(region),*placement[:4])
                    canvas.text(x+cell_w/2,y+cell_h/2,"CAM INVALID",size=6,anchor="middle",color="#9C2020")
                    continue
                normalizations.append(dict(model_id=mid,image=key,full_image_normalization=normalization,
                    alpha=config.get("overlay_alpha",.45),colormap="JET-256-linear-segments-v1",interpolation="Pillow BILINEAR half-pixel centers",
                    shared_view_region=region,native_cam_npz_sha256=sha256(entry["npz"])))
            canvas.image(panel.crop(region),*placement[:4])
            if config.get("full_panel_dir"):
                directory=Path(config["full_panel_dir"])
                if heatmap and i>0:
                    path=directory/"heatmaps"/models[i-1]["id"]/(Path(key).stem+".jpg")
                    path.parent.mkdir(parents=True,exist_ok=True)
                    panel.save(path,quality=90,subsampling=0,optimize=True)
            if not heatmap:
                boxes=images[key]["annotations"] if i==0 else [p for p in predictions[models[i-1]["id"]][key] if p["score"]>=case["display_threshold"]]
                boxes_on_canvas(canvas,boxes,region,placement,GT_COLOR if i==0 else PRED_COLOR,scores=i>0)
                if config.get("full_panel_dir"):
                    save_full_detection(original,boxes,Path(config["full_panel_dir"])/"full_panels"/("ground_truth" if i==0 else models[i-1]["id"])/(Path(key).stem+".jpg"),scores=i>0)
    if heatmap:
        bar_x=margin+label;bar_y=bottom+10;bar_w=90
        lut=Image.fromarray(np.round(jet(np.linspace(0,1,256)[None,:])*255).astype(np.uint8))
        canvas.image(lut,bar_x,bar_y,bar_w,5)
        canvas.text(bar_x-3,bar_y+5,"0",size=6,anchor="end")
        canvas.text(bar_x+bar_w+3,bar_y+5,"1",size=6)
        canvas.text(bar_x+bar_w+14,bar_y+5,"Relative response (per-map 0–1)",size=6)
    status=("draft_pending_cam" if heatmap else "draft_pending_predictions") if pending else "draft_pending_visual_review" if cases_doc["status"]!="reviewed" else "reviewed"
    if status!="reviewed":
        canvas.text(width-margin,height-2,"DRAFT — "+status.replace("_"," "),size=6,anchor="end",color="#9C2020")
    record=canvas.export(output,formats)
    record.update(status=status,row_labels=labels,column_images=[c["image"] for c in cases],
                  rows=len(labels),columns=len(cases),pending=pending,cases_status=cases_doc["status"],
                  heatmap_display_normalizations=normalizations,renderer_sha256=sha256(__file__))
    write_json(Path(output).with_suffix(".layout.json"),record)
    return record


def candidate_previews(document, images, predictions, models, image_root, output, config):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for case in document["cases"]:
        single={**document,"cases":[case]}
        matrix(single,images,predictions,models,image_root,output/(Path(case["image"]).stem),
               config={**config,"width_inches":3.4,"max_height_inches":12,"dpi":140},draft=True,formats=("png",))
    # Grouped GT contact sheets keep the review package light and make the ranking auditable.
    for group in sorted({c["advantage_type"] for c in document["cases"]}):
        rows=[c for c in document["cases"] if c["advantage_type"]==group]
        for start in range(0,len(rows),8):
            page=rows[start:start+8];canvas=Canvas(570,430,150)
            canvas.text(10,15,group+" — pending visual review",size=10)
            for i,case in enumerate(page):
                x=10+(i%4)*140;y=25+(i//4)*200
                with Image.open(Path(image_root)/Path(case["image"]).name) as im:original=im.convert("RGB")
                placement=fitted(case["view_region"],x,y,132,150)
                canvas.image(original,*placement[:4])
                boxes_on_canvas(canvas,images[case["image"]]["annotations"],case["view_region"],placement,GT_COLOR)
                canvas.text(x,y+160,f"Candidate {case['order']}: {Path(case['image']).stem[:23]}",size=6)
                canvas.text(x,y+170,"strong-model deltas: "+str(tuple(case["comparison_error_delta"][m] for m in ("rtdetr_c2","yolo26m","yolov13l"))),size=6)
                canvas.text(x,y+180,"Verify image/GT, scene and duplicates",size=6)
            canvas.export(output/f"contact_{group}_{start//8+1}",formats=("png",))
