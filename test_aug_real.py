import cv2, numpy as np, random
from pathlib import Path
import albumentations as A

COLORS = {0:(255,255,255),1:(0,255,0),2:(0,150,255),3:(255,165,0),
          4:(255,50,50),5:(255,180,50),6:(255,255,50),7:(50,255,50),
          8:(50,255,220),9:(50,220,255),10:(80,80,255),11:(220,50,255),
          12:(255,50,200),13:(200,200,100)}
CLASS_NAME = {0:'etiket',1:'TR',2:'logo',3:'QR',
              4:'0',5:'1',6:'2',7:'3',8:'4',9:'5',10:'6',11:'7',12:'8',13:'9'}

def yolo_to_px(lines, W, H):
    result = []
    for line in lines:
        if not line.strip(): continue
        parts = line.split()
        cid = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = [(coords[i]*W, coords[i+1]*H) for i in range(0, len(coords), 2)]
        result.append((cid, pts))
    return result

def augment(img_bgr, ann, seed):
    rng = np.random.RandomState(seed)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # Dikey flip
    if rng.random() < 0.5:
        img = img[::-1].copy()
        ann = [(cid, [(x, H-1-y) for x,y in pts]) for cid,pts in ann]

    # Kucuk rotasyon +-5
    angle = rng.uniform(-5, 5)
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (W,H), borderMode=cv2.BORDER_REFLECT_101)
        new_ann = []
        for cid, pts in ann:
            new_pts = []
            for x, y in pts:
                nx = M[0,0]*x + M[0,1]*y + M[0,2]
                ny = M[1,0]*x + M[1,1]*y + M[1,2]
                new_pts.append((max(0.0, min(W-1, nx)), max(0.0, min(H-1, ny))))
            new_ann.append((cid, new_pts))
        ann = new_ann

    # Photometric
    img = A.Compose([
        A.RandomBrightnessContrast(0.12, 0.12, p=0.7),
        A.HueSaturationValue(5, 12, 10, p=0.6),
        A.RandomGamma((90,115), p=0.4),
        A.GaussNoise(var_limit=(2,8), p=0.5),
        A.MotionBlur(blur_limit=(3,5), p=0.1),
        A.ImageCompression(quality_lower=75, quality_upper=98, p=0.4),
    ])(image=img)['image']

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), ann

def draw(img, ann):
    ov = img.copy()
    H, W = img.shape[:2]
    for cid, pts in ann:
        npts = np.array([[max(0,min(W-1,int(x))), max(0,min(H-1,int(y)))] for x,y in pts], dtype=np.int32)
        color = COLORS.get(cid,(255,255,255))
        cv2.polylines(ov, [npts], True, (color[2],color[1],color[0]), 4)
        cx,cy = int(np.mean(npts[:,0])), int(np.mean(npts[:,1]))
        cv2.putText(ov, CLASS_NAME.get(cid,'?'), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (color[2],color[1],color[0]), 2)
    return ov

# --- Crop ornegi ---
crop_img_path = Path('real_aug2/crops/images/usb_frame_20260622_103624_143_jpg.rf.zPaTsTohjjr9fX4Otnwv_etiket_00.jpg')
crop_lbl_path = Path('real_aug2/crops/labels/usb_frame_20260622_103624_143_jpg.rf.zPaTsTohjjr9fX4Otnwv_etiket_00.txt')
crop_img = cv2.imread(str(crop_img_path))
H,W = crop_img.shape[:2]
crop_ann = yolo_to_px(crop_lbl_path.read_text().strip().split('\n'), W, H)

panels = []
for seed in range(10):
    aug_img, aug_ann = augment(crop_img, crop_ann, seed * 7 + 42)
    panels.append(cv2.resize(draw(aug_img, aug_ann), (500,500)))

grid = np.vstack([np.hstack(panels[:5]), np.hstack(panels[5:])])
cv2.imwrite('test_aug_real.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
print('ok - test_aug_real.jpg')
