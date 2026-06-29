import cv2, numpy as np
from pathlib import Path

IMG_DIR = Path('real_aug/images')
LBL_DIR = Path('real_aug/labels')

COLORS = {0:(200,200,200),1:(0,200,0),2:(0,100,255),3:(255,165,0),
          4:(255,50,50),5:(255,150,50),6:(255,230,50),7:(50,255,50),
          8:(50,255,200),9:(50,200,255),10:(50,50,255),11:(200,50,255),
          12:(255,50,200),13:(180,180,100)}

imgs = sorted(IMG_DIR.glob('*.jpg'))
samples = [imgs[i] for i in [0, 100, 250, 400, 550, 700]]

thumbs = []
for img_path in samples:
    lbl_path = LBL_DIR / (img_path.stem + '.txt')
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split('\n'):
            if not line: continue
            parts = line.split()
            cid = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array([[int(coords[i]*W), int(coords[i+1]*H)]
                           for i in range(0, len(coords), 2)], dtype=np.int32)
            color = COLORS.get(cid, (255,255,255))
            cv2.polylines(img, [pts], True, (color[2],color[1],color[0]), 2)

    thumbs.append(cv2.resize(img, (700, 525)))

grid = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:])])
cv2.imwrite('preview_real_check.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok')
