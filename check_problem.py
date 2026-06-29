import cv2, numpy as np
from pathlib import Path

IMG_DIR = Path('real_aug/images')
LBL_DIR = Path('real_aug/labels')

COLORS = {0:(255,255,255),1:(0,255,0),2:(0,150,255),3:(255,165,0),
          4:(255,50,50),5:(255,180,50),6:(255,255,50),7:(50,255,50),
          8:(50,255,220),9:(50,220,255),10:(80,80,255),11:(220,50,255),
          12:(255,50,200),13:(200,200,100)}
CLASS_NAME = {0:'etiket',1:'TR',2:'logo',3:'QR',
              4:'r0',5:'r1',6:'r2',7:'r3',8:'r4',9:'r5',10:'r6',11:'r7',12:'r8',13:'r9'}

e03 = sorted(IMG_DIR.glob('*etiket_03*'))
e02 = sorted(IMG_DIR.glob('*etiket_02*'))
targets = [e03[0], e03[10], e03[20], e02[0], e02[10], e02[20]]

out_panels = []
for img_path in targets:
    lbl_path = LBL_DIR / (img_path.stem + '.txt')
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    ov = img.copy()
    classes_found = []
    n_annot = 0
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split('\n'):
            if not line: continue
            parts = line.split()
            cid = int(parts[0])
            classes_found.append(cid)
            n_annot += 1
            coords = list(map(float, parts[1:]))
            pts = np.array([[int(coords[i]*W), int(coords[i+1]*H)]
                           for i in range(0, len(coords), 2)], dtype=np.int32)
            color = COLORS.get(cid, (255,255,255))
            bgr = (color[2], color[1], color[0])
            cv2.polylines(ov, [pts], True, bgr, 4)
            cx_pt = int(np.mean(pts[:,0]))
            cy_pt = int(np.mean(pts[:,1]))
            cv2.putText(ov, CLASS_NAME.get(cid,'?'), (cx_pt,cy_pt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    info = f'{img_path.stem[-25:]}  {n_annot}ann  {sorted(set(classes_found))}'
    cv2.putText(ov, info, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    out_panels.append(cv2.resize(ov, (900, 675)))

grid = np.vstack([np.hstack(out_panels[:3]), np.hstack(out_panels[3:])])
cv2.imwrite('preview_problem.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
print('ok - preview_problem.jpg')
