import json, glob, tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from PIL import Image, ImageTk

CONFIG_DIR = Path('configs')

def load_configs():
    return sorted(CONFIG_DIR.glob('*.json'))

class EtiketEditor:
    def __init__(self, root):
        self.root = root
        self.root.title('Etiket Editor')
        self.configs = load_configs()
        self.cfg = {}
        self.show_qr = tk.BooleanVar(value=True)
        self._img_ref = None

        # Sol panel: scrollable kontroller
        left_outer = tk.Frame(root, width=350, bg='#2b2b2b')
        left_outer.pack(side='left', fill='y')
        left_outer.pack_propagate(False)

        canvas_scroll = tk.Canvas(left_outer, bg='#2b2b2b', highlightthickness=0, width=340)
        scrollbar = tk.Scrollbar(left_outer, orient='vertical', command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas_scroll.pack(side='left', fill='both', expand=True)

        left = tk.Frame(canvas_scroll, bg='#2b2b2b')
        canvas_scroll.create_window((0, 0), window=left, anchor='nw')
        left.bind('<Configure>', lambda e: canvas_scroll.configure(
            scrollregion=canvas_scroll.bbox('all')))
        canvas_scroll.bind('<MouseWheel>',
            lambda e: canvas_scroll.yview_scroll(-1*(e.delta//120), 'units'))

        # Config secici
        cfg_row = tk.Frame(left, bg='#2b2b2b')
        cfg_row.pack(fill='x', padx=5, pady=2)
        tk.Label(cfg_row, text='Config:', bg='#2b2b2b', fg='white').pack(side='left')
        tk.Button(cfg_row, text='Ac...', bg='#555', fg='white', padx=6,
                  command=self.open_file).pack(side='right')

        self.cfg_var = tk.StringVar()
        cfg_names = [p.name for p in self.configs]
        self.cfg_combo = ttk.Combobox(left, textvariable=self.cfg_var, values=cfg_names, width=35)
        self.cfg_combo.pack(padx=5, pady=2)
        self.cfg_combo.bind('<<ComboboxSelected>>', self.on_config_change)

        tk.Checkbutton(left, text='QR Goster', variable=self.show_qr,
                       bg='#2b2b2b', fg='white', selectcolor='#444',
                       command=self.update_preview).pack(anchor='w', padx=5)

        # Numara girisi
        num_row = tk.Frame(left, bg='#2b2b2b')
        num_row.pack(fill='x', padx=5, pady=2)
        tk.Label(num_row, text='Numara:', bg='#2b2b2b', fg='white').pack(side='left')
        self.num_var = tk.StringVar(value='12345678')
        num_entry = tk.Entry(num_row, textvariable=self.num_var, width=12,
                             bg='#444', fg='white', insertbackground='white')
        num_entry.pack(side='left', padx=4)
        num_entry.bind('<KeyRelease>', lambda e: self.update_preview())
        # 7 / 8 hane butonlari
        tk.Button(num_row, text='7h', bg='#555', fg='white', padx=4,
                  command=lambda: [self.num_var.set('1234567'), self.update_preview()]
                  ).pack(side='left', padx=1)
        tk.Button(num_row, text='8h', bg='#555', fg='white', padx=4,
                  command=lambda: [self.num_var.set('12345678'), self.update_preview()]
                  ).pack(side='left', padx=1)

        # Slider grupları
        self.sliders = {}
        groups = [
            ('Pin / Sekil', [
                ('pin_r',      20, 200),
                ('pin_cy',      0, 200),
                ('shaft_w',    20, 400),
                ('shaft_bot',  50, 400),
                ('shoulder_y',100, 700),
                ('top_corner_r', 0, 150),
            ]),
            ('TR', [('tr_cx', 0, 785), ('tr_cy', 0, 900), ('tr_h', 20, 300)]),
            ('Logo', [('logo_cx', 0, 785), ('logo_cy', 0, 900),
                      ('logo_w', 20, 300), ('logo_h', 20, 300)]),
            ('QR', [('qr_cx', 0, 785), ('qr_cy', 0, 900),
                    ('qr_w', 20, 300), ('qr_h', 20, 300)]),
            ('Numara', [('num_cx', 0, 785), ('num_cy', 0, 900), ('num_h', 20, 300)]),
        ]

        for group, params in groups:
            lf = tk.LabelFrame(left, text=group, bg='#2b2b2b', fg='#aaa',
                               labelanchor='nw', padx=4, pady=2)
            lf.pack(fill='x', padx=5, pady=3)
            for key, mn, mx in params:
                row = tk.Frame(lf, bg='#2b2b2b')
                row.pack(fill='x')
                tk.Label(row, text=f'{key:<12}', bg='#2b2b2b', fg='white',
                         font=('Courier', 9), width=12, anchor='w').pack(side='left')
                val_lbl = tk.Label(row, text='0', bg='#2b2b2b', fg='#ff0',
                                   font=('Courier', 9), width=5)
                val_lbl.pack(side='right')
                s = tk.Scale(row, from_=mn, to=mx, orient='horizontal',
                             bg='#2b2b2b', fg='white', troughcolor='#555',
                             highlightthickness=0, bd=0, sliderlength=12,
                             command=lambda v, k=key: self.on_slider(k, v))
                s.pack(side='left', fill='x', expand=True)
                self.sliders[key] = (s, val_lbl)

        # Kaydet butonu
        tk.Button(left, text='Kaydet (JSON)', bg='#2a6', fg='white',
                  font=('Arial', 10, 'bold'),
                  command=self.save_config).pack(fill='x', padx=5, pady=6)

        # Sag panel: preview
        right = tk.Frame(root, bg='#1a1a1a')
        right.pack(side='right', fill='both', expand=True)
        self.canvas = tk.Canvas(right, bg='#1a1a1a', cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.update_preview())

        # Ilk config yukle
        if self.configs:
            self.cfg_combo.set(self.configs[0].name)
            self.on_config_change()

    def load_path(self, path):
        path = Path(path)
        if not path.exists():
            return
        self._current_path = path
        with open(path) as f:
            self.cfg = json.load(f)
        name = path.name
        self.show_qr.set('tip2' in name)
        self.num_var.set(self.cfg.get('number', '12345678'))
        for key, (slider, lbl) in self.sliders.items():
            v = self.cfg.get(key, 0)
            slider.set(v)
            lbl.config(text=str(v))
        self.root.title(f'Etiket Editor — {name}')
        self.update_preview()

    def on_config_change(self, event=None):
        name = self.cfg_var.get()
        self.load_path(CONFIG_DIR / name)

    def open_file(self):
        path = filedialog.askopenfilename(
            title='JSON sec',
            initialdir=str(CONFIG_DIR),
            filetypes=[('JSON', '*.json'), ('Tum', '*.*')])
        if path:
            self.cfg_var.set(Path(path).name)
            self.load_path(path)

    def on_slider(self, key, val):
        if not self.cfg:
            return
        v = int(float(val))
        self.cfg[key] = v
        s, lbl = self.sliders[key]
        lbl.config(text=str(v))
        self.update_preview()

    def update_preview(self):
        if not self.cfg:
            return
        try:
            from etiket_editor import render, FONT_CACHE, _LOGO_CACHE
            # Cache temizle ki degisiklikler yansisin
            _LOGO_CACHE.clear()
            number = self.num_var.get() or self.cfg.get('number', '12345678')
            img = render(self.cfg, show_qr=self.show_qr.get(), number=number)

            cw = self.canvas.winfo_width() or 500
            ch = self.canvas.winfo_height() or 700
            scale = min(cw / img.width, ch / img.height) * 0.95
            nw = int(img.width * scale)
            nh = int(img.height * scale)
            img_r = img.resize((nw, nh), Image.LANCZOS)
            self._img_ref = ImageTk.PhotoImage(img_r)
            self.canvas.delete('all')
            self.canvas.create_image(cw//2, ch//2, anchor='center', image=self._img_ref)
        except Exception as e:
            self.canvas.delete('all')
            self.canvas.create_text(10, 10, anchor='nw', text=str(e), fill='red')

    def save_config(self):
        path = getattr(self, '_current_path', None)
        if not path:
            path = filedialog.asksaveasfilename(
                title='Kaydet',
                initialdir=str(CONFIG_DIR),
                defaultextension='.json',
                filetypes=[('JSON', '*.json')])
            if not path:
                return
        self.cfg['number'] = self.num_var.get() or self.cfg.get('number', '12345678')
        with open(path, 'w') as f:
            json.dump(self.cfg, f, indent=2)
        self.canvas.create_text(10, 10, anchor='nw',
                                text=f'Kaydedildi: {Path(path).name}', fill='#0f0', font=('Arial', 11))
        self.root.after(2000, self.update_preview)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1100x780')
    root.configure(bg='#1a1a1a')
    app = EtiketEditor(root)
    root.mainloop()
