import json, random, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import qrcode as qrc

def _bezier2(p0,p1,p2,n=50):
    t=np.linspace(0,1,n)
    x=(1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t**2*p2[0]
    y=(1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t**2*p2[1]
    return list(zip(x.astype(int),y.astype(int)))

def _bezier3(p0,p1,p2,p3,n=80):
    t=np.linspace(0,1,n)
    x=(1-t)**3*p0[0]+3*(1-t)**2*t*p1[0]+3*(1-t)*t**2*p2[0]+t**3*p3[0]
    y=(1-t)**3*p0[1]+3*(1-t)**2*t*p1[1]+3*(1-t)*t**2*p2[1]+t**3*p3[1]
    return list(zip(x.astype(int),y.astype(int)))

def _fit_font(text, max_h, path='arial.ttf'):
    tmp=ImageDraw.Draw(Image.new('L',(2000,2000)))
    for fs in range(500,5,-1):
        f=ImageFont.truetype(path,fs)
        bb=tmp.textbbox((0,0),text,font=f)
        if (bb[3]-bb[1])<=max_h: return f
    return ImageFont.truetype(path,10)

FONT_CACHE = {}

def _fit_font_cached(text, max_h, path='arial.ttf'):
    key = (max_h, path)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = _fit_font(text, max_h, path)
    return FONT_CACHE[key]

DIGIT_COLORS = {
    '0':(255,50,50),'1':(255,150,50),'2':(255,230,50),'3':(50,255,50),
    '4':(50,255,200),'5':(50,200,255),'6':(50,50,255),'7':(200,50,255),
    '8':(255,50,200),'9':(180,180,100)
}
W_IMG = 785
LOGO_CACHE = {}

def render_pair(cfg, show_qr, number):
    from etiket_editor import render as render_img
    img = render_img(cfg, show_qr=show_qr, number=number)

    pad=cfg['top_pad']; H=847+pad
    bm=cfg['body_margin']; cr=cfg['corner_r']; tcr=cfg['top_corner_r']
    sh=cfg['shaft_w']//2; sy=cfg['shoulder_y']+pad; sb=cfg['shaft_bot']+pad
    bb_=cfg['body_bot']+pad; cx=W_IMG//2; pcy=cfg['pin_cy']+pad; pr=cfg['pin_r']
    tc=tuple(int(cfg['text_color'].lstrip('#')[i:i+2],16) for i in (0,2,4))
    lc=tuple(int(cfg['label_color'].lstrip('#')[i:i+2],16) for i in (0,2,4))
    import math
    t_len = max(70, int((sy - sb) * 0.55))

    # Pin arc: sag (0°) → sol (180°), ust yari daire
    arc_pts = []
    for a in range(0, 181, 3):
        rad = math.radians(a)
        arc_pts.append((int(cx + pr * math.cos(rad)),
                        int(pcy - pr * math.sin(rad))))

    m=np.zeros((H,W_IMG,3),dtype=np.uint8)
    pts=[]
    sol=(bm+tcr,sy)
    pts+=_bezier3((cx-sh,sb),(cx-sh,sb+t_len),(sol[0]+t_len,sol[1]),sol)
    if tcr>0: pts+=_bezier2((bm+tcr,sy),(bm,sy),(bm,sy+tcr),n=25)
    pts.append((bm,bb_-cr)); pts+=_bezier2((bm,bb_-cr),(bm,bb_),(bm+cr,bb_),n=20)
    pts.append((W_IMG-bm-cr,bb_)); pts+=_bezier2((W_IMG-bm-cr,bb_),(W_IMG-bm,bb_),(W_IMG-bm,bb_-cr),n=20)
    pts.append((W_IMG-bm,sy+tcr if tcr>0 else sy))
    if tcr>0: pts+=_bezier2((W_IMG-bm,sy+tcr),(W_IMG-bm,sy),(W_IMG-bm-tcr,sy),n=20)
    sag=(W_IMG-bm-tcr,sy)
    pts+=_bezier3(sag,(sag[0]-t_len,sag[1]),(cx+sh,sb+t_len),(cx+sh,sb))
    pts.append((cx+sh, pcy))
    pts+=arc_pts
    pts_cv=np.array(pts,dtype=np.int32)
    cv2.fillPoly(m,[pts_cv],(200,200,200))
    mask=Image.fromarray(m)

    # etiket_editor.render() ile ayni hizalama: TR/logo/QR icin row_cy kullan
    row_cy = round((cfg['tr_cy'] + cfg['logo_cy'] + cfg['qr_cy']) / 3) + pad

    # TR piksel-level
    ft=_fit_font_cached('TR',cfg['tr_h'])
    d=ImageDraw.Draw(Image.new('L',(1,1)))
    bt=d.textbbox((0,0),'TR',font=ft)
    tx=cfg['tr_cx']-(bt[2]-bt[0])//2; ty=row_cy-(bt[3]-bt[1])//2-bt[1]
    tmp=Image.new('L',(W_IMG,H),0)
    ImageDraw.Draw(tmp).text((tx,ty),'TR',fill=255,font=ft)
    m[np.array(tmp)>128]=[0,200,0]

    # Logo - dolu ellipse (logo_w x logo_h tam boyut)
    lcx=cfg['logo_cx']; lcy=row_cy
    cv2.ellipse(m,(lcx,lcy),(cfg['logo_w']//2,cfg['logo_h']//2),0,0,360,[0,100,255],-1)

    # QR - dolu dikdortgen (basit)
    if show_qr:
        qx=min(cfg['qr_cx']-cfg['qr_w']//2, W_IMG-bm-cfg['qr_w']); qy_=row_cy-cfg['qr_h']//2
        y0q=max(0,qy_); y1q=min(H,qy_+cfg['qr_h']); x0q=max(0,qx); x1q=min(W_IMG,qx+cfg['qr_w'])
        m[y0q:y1q,x0q:x1q]=[255,165,0]

    # Rakamlar piksel-level — etiket_editor ile ayni font + pozisyon
    from etiket_editor import _fit_font_cached as _ec_fit
    max_nw=int((W_IMG-2*bm)*cfg.get('num_max_w_pct',98)/100)
    fn=_ec_fit(number,cfg['num_h'],max_nw)   # height+width constrained, etiket_editor ile ayni
    bn=d.textbbox((0,0),number,font=fn)
    nw=bn[2]-bn[0]; nh=bn[3]-bn[1]
    nx0=max(bm, min(W_IMG-bm-nw, cfg['num_cx']-nw//2-bn[0]))
    ny =cfg['num_cy']+pad-nh//2-bn[1]
    # Tum number icin maske temp image
    tf=Image.new('L',(nw+10,nh+10),0)
    ImageDraw.Draw(tf).text((-bn[0],-bn[1]),number,fill=255,font=fn)
    for i,dig in enumerate(number):
        col=DIGIT_COLORS[dig]
        bp=d.textbbox((0,0),number[:i],font=fn) if i>0 else (0,0,0,0)
        bc=d.textbbox((0,0),number[:i+1],font=fn)
        x0c=max(0,(bp[2] if i>0 else 0)-bn[0])
        x1c=min(nw+10, bc[2]-bn[0])
        if x1c<=x0c: continue
        dc=tf.crop((x0c,0,x1c,nh+10))
        px=nx0+x0c
        ca=np.array(dc); hc,wc=ca.shape
        y0p=ny+bn[1]; y1p=min(ny+bn[1]+hc,H); x0p=px; x1p=min(px+wc,W_IMG)
        if y0p>=H or x0p>=W_IMG or y1p<=0 or x1p<=0: continue
        cy0=max(0,-y0p); cy1=cy0+(y1p-max(0,y0p))
        cx0_=max(0,-x0p); cx1_=cx0_+(x1p-max(0,x0p))
        iy0=max(0,y0p); ix0=max(0,x0p)
        reg=ca[cy0:cy1,cx0_:cx1_]
        m[iy0:iy0+reg.shape[0],ix0:ix0+reg.shape[1]][reg>128]=list(col)

    return img, Image.fromarray(m)


def _add_lighting(img_np, rng):
    H, W = img_np.shape[:2]
    out = img_np.astype(np.float32) * rng.uniform(0.80, 1.20)
    for _ in range(rng.randint(1, 3)):
        Y, X = np.ogrid[:H, :W]
        corner = rng.choice(['TL','TR','BL','BR','T','B','L','R'])
        if   corner=='TL': d=(X/W+Y/H)/2.0
        elif corner=='TR': d=((W-X)/W+Y/H)/2.0
        elif corner=='BL': d=(X/W+(H-Y)/H)/2.0
        elif corner=='BR': d=((W-X)/W+(H-Y)/H)/2.0
        elif corner=='T':  d=Y/float(H)
        elif corner=='B':  d=(H-Y)/float(H)
        elif corner=='L':  d=X/float(W)
        else:              d=(W-X)/float(W)
        falloff=rng.uniform(0.4,0.85)
        shadow=np.clip(1.0-d/falloff,0,1)
        ks=int(rng.uniform(W//8,W//4))|1
        shadow=cv2.GaussianBlur(shadow.astype(np.float32),(ks,ks),0)
        out=out*(1.0-shadow[:,:,np.newaxis]*rng.uniform(0.25,0.55))
    return np.clip(out,0,255).astype(np.uint8)


def augment_pair(img_pil, mask_pil, angle, seed=None):
    import io
    if seed is not None: random.seed(seed); np.random.seed(seed)
    rng=np.random.RandomState(seed if seed is not None else random.randint(0,99999))

    BG=(0xd1,0xc4,0x26); PAD=250
    W_bg=img_pil.width+2*PAD; H_bg=img_pil.height+2*PAD
    bg=Image.new('RGB',(W_bg,H_bg),BG)
    ox=PAD+random.randint(-15,15); oy=PAD+random.randint(-15,15)
    bg.paste(img_pil,(ox,oy))
    mb=Image.new('RGB',(W_bg,H_bg),(0,0,0))
    mb.paste(mask_pil,(ox,oy))

    img_np=cv2.cvtColor(np.array(bg),cv2.COLOR_RGB2BGR)
    msk_np=cv2.cvtColor(np.array(mb),cv2.COLOR_RGB2BGR)

    # --- Geometrik (maske de ayni transform, INTER_NEAREST) ---
    def rnd(): return int(random.uniform(-0.03,0.03)*min(W_bg,H_bg))
    src=np.float32([[0,0],[W_bg,0],[W_bg,H_bg],[0,H_bg]])
    dst=np.float32([[rnd(),rnd()],[W_bg+rnd(),rnd()],[W_bg+rnd(),H_bg+rnd()],[rnd(),H_bg+rnd()]])
    M_p=cv2.getPerspectiveTransform(src,dst)
    scale=float(rng.uniform(0.75,1.10))
    M_r=cv2.getRotationMatrix2D((W_bg//2,H_bg//2),angle,scale)

    img_np=cv2.warpPerspective(img_np,M_p,(W_bg,H_bg),borderMode=cv2.BORDER_REPLICATE)
    img_np=cv2.warpAffine(img_np,M_r,(W_bg,H_bg),borderMode=cv2.BORDER_REPLICATE)
    msk_np=cv2.warpPerspective(msk_np,M_p,(W_bg,H_bg),
                               flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    msk_np=cv2.warpAffine(msk_np,M_r,(W_bg,H_bg),
                          flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    # --- Fotometrik (sadece gorsel) ---
    photo=Image.fromarray(cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB))

    # Blur
    photo=photo.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3,1.5)))
    # Brightness + Contrast
    photo=ImageEnhance.Brightness(photo).enhance(random.uniform(0.75,1.3))
    photo=ImageEnhance.Contrast(photo).enhance(random.uniform(0.85,1.2))
    # Saturation + Hue (HSV)
    hsv=cv2.cvtColor(np.array(photo),cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,0]=(hsv[:,:,0]+rng.uniform(-5,5))%180
    hsv[:,:,1]=np.clip(hsv[:,:,1]*rng.uniform(0.85,1.15),0,255)
    photo=Image.fromarray(cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8),cv2.COLOR_HSV2RGB))
    # Gaussian noise
    noise=np.random.normal(0,random.uniform(2,5),np.array(photo).shape).astype(np.int16)
    photo=Image.fromarray(np.clip(np.array(photo).astype(np.int16)+noise,0,255).astype(np.uint8))
    # JPEG compression artifact (%30 ihtimalle)
    if rng.random()<0.30:
        buf=io.BytesIO()
        photo.save(buf,format='JPEG',quality=int(rng.uniform(40,70)))
        buf.seek(0); photo=Image.open(buf).copy()

    # --- Isik / Golge (sadece gorsel) ---
    if rng.random()<0.80:
        photo=Image.fromarray(_add_lighting(np.array(photo),rng))

    mask_out=Image.fromarray(cv2.cvtColor(msk_np,cv2.COLOR_BGR2RGB))
    return photo, mask_out


CC = {0:(200,200,200),1:(0,200,0),2:(0,100,255),3:(255,165,0),4:(255,50,50),
      5:(255,150,50),6:(255,230,50),7:(50,255,50),8:(50,255,200),9:(50,200,255),
      10:(50,50,255),11:(200,50,255),12:(255,50,200),13:(180,180,100)}

def verify_overlay(aug_img, aug_msk):
    img_cv=cv2.cvtColor(np.array(aug_img),cv2.COLOR_RGB2BGR)
    msk_np=np.array(aug_msk)
    ov=img_cv.copy()
    for cid,color in CC.items():
        min_a=50
        mode=cv2.RETR_EXTERNAL
        c=np.array(color,dtype=np.int32)
        diff=np.abs(msk_np.astype(np.int32)-c).max(axis=2)
        binary=(diff<5).astype(np.uint8)*255
        cnts,_=cv2.findContours(binary,mode,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt)<min_a: continue
            eps=0.002*cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,eps,True)
            cv2.polylines(ov,[approx],True,(color[2],color[1],color[0]),2)
    return ov


if __name__ == '__main__':
    random.seed(42)
    with open('configs/tip2_0001.json') as f:
        cfg=json.load(f)
    number=cfg.get('number','12345678')

    img, msk = render_pair(cfg, show_qr=True, number=number)
    aug_img, aug_msk = augment_pair(img, msk, angle=15)
    aug_img.save('sample_aug_img.jpg', quality=92)
    aug_msk.save('sample_aug_mask.png')
    ov = verify_overlay(aug_img, aug_msk)
    cv2.imwrite('sample_aug_verify.png', ov)
    print('number:', number)
    print('Tamam')
