import cv2, numpy as np
from pathlib import Path

IMG_DIR = Path('real_aug/images')
LBL_DIR = Path('real_aug/labels')

COLORS = {0:(255,255,255),1:(0,255,0),2:(0,150,255),3:(255,165,0),
          4:(255,50,50),5:(255,180,50),6:(255,255,50),7:(50,255,50),
          8:(50,255,220),9:(50,220,255),10:(80,80,255),11:(220,50,255),
          12:(255,50,200),13:(200,200,100)}
CLASS_NAME = {0:'etiket',1:'TR',2:'logo',3:'QR',
              4:'0',5:'1',6:'2',7:'3',8:'4',9:'5',10:'6',11:'7',12:'8',13:'9'}

imgs = sorted(IMG_DIR.glob('wide_usb_frame*'))
samples = [imgs[0], imgs[15], imgs[30], imgs[45]]

panels = []
for img_path in samples:
    lbl_path = LBL_DIR / (img_path.stem + '.txt')
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    ov = img.copy()

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split('\n'):
            if not line: continue
            parts = line.split()
            cid = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array([[int(coords[i]*W), int(coords[i+1]*H)]
                           for i in range(0, len(coords), 2)], dtype=np.int32)
            color = COLORS.get(cid, (255,255,255))
            bgr = (color[2], color[1], color[0])
            cv2.polylines(ov, [pts], True, bgr, 5)
            cx_pt = int(np.mean(pts[:,0]))
            cy_pt = int(np.mean(pts[:,1]))
            cv2.putText(ov, CLASS_NAME.get(cid,'?'), (cx_pt, cy_pt),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr, 3)

    cv2.putText(ov, img_path.stem[-20:], (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    panels.append(cv2.resize(ov, (800, 600)))

grid = np.hstack(panels)
cv2.imwrite('check_wide.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
print('ok - check_wide.jpg')
