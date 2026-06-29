import cv2, numpy as np
from pathlib import Path

IMG_DIR = Path('real_aug/images')
LBL_DIR = Path('real_aug/labels')

COLORS = {0:(200,200,200),1:(0,200,0),2:(0,100,255),3:(255,165,0),
          4:(255,50,50),5:(255,150,50),6:(255,230,50),7:(50,255,50),
          8:(50,255,200),9:(50,200,255),10:(50,50,255),11:(200,50,255),
          12:(255,50,200),13:(180,180,100)}

CLASS_NAME = {0:'etiket',1:'TR',2:'logo',3:'QR',
              4:'0',5:'1',6:'2',7:'3',8:'4',9:'5',10:'6',11:'7',12:'8',13:'9'}

imgs = sorted(IMG_DIR.glob('*.jpg'))
# 6 farkli orijinal etiketten birer tane sec
samples = [imgs[0], imgs[50], imgs[150], imgs[300], imgs[500], imgs[700]]

out_panels = []
for img_path in samples:
    lbl_path = LBL_DIR / (img_path.stem + '.txt')
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    ov = img.copy()

    classes_found = []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split('\n'):
            if not line: continue
            parts = line.split()
            cid = int(parts[0])
            classes_found.append(CLASS_NAME.get(cid, str(cid)))
            coords = list(map(float, parts[1:]))
            pts = np.array([[int(coords[i]*W), int(coords[i+1]*H)]
                           for i in range(0, len(coords), 2)], dtype=np.int32)
            color = COLORS.get(cid, (255,255,255))
            cv2.polylines(ov, [pts], True, (color[2],color[1],color[0]), 3)
            # Label yaz
            cx_pt = int(np.mean(pts[:,0]))
            cy_pt = int(np.mean(pts[:,1]))
            cv2.putText(ov, CLASS_NAME.get(cid,'?'), (cx_pt,cy_pt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (color[2],color[1],color[0]), 2)

    blended = cv2.addWeighted(img, 0.4, ov, 0.6, 0)

    # Boyut bilgisi + dosya adi
    info = f'{img_path.name[:40]}  {W}x{H}'
    classes_str = ' '.join(sorted(set(classes_found)))
    cv2.putText(blended, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(blended, classes_str, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    # 800x600'e boyutlandir
    out_panels.append(cv2.resize(blended, (800, 600)))

grid = np.vstack([np.hstack(out_panels[:3]), np.hstack(out_panels[3:])])
cv2.imwrite('preview_real2.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
print('ok')
