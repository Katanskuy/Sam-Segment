import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys

from ultralytics import FastSAM
from PIL import Image, ImageTk


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


model_path = resource_path("FastSAM-s.pt")
model = FastSAM(model_path)


class StoneCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Stone Counter Pro v7.0")
        self.root.geometry("800x750")
        self.root.configure(bg="#ffffff")

        # Налаштування чутливості
        self.CONF_DEFAULT = 0.12
        self.AREA_DEFAULT = 30
        self.OVERLAP_DEFAULT = 0.3

        self.setup_ui()
        self.last_result = None
        self.tk_image = None
        self.original_filename = ""
        self.path = ""

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#ffffff")
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=20)

        self.header = tk.Label(self.main_frame, text="Аналіз зображень (Deep Scan)",
                               font=("Segoe UI", 20, "bold"), bg="#ffffff", fg="#2d3436")
        self.header.pack(pady=(0, 5))

        self.status_card = tk.Frame(self.main_frame, bg="#f8f9fa", bd=0,
                                    highlightbackground="#dfe4ea", highlightthickness=1)
        self.status_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.result_display = tk.Label(self.status_card, bg="#f8f9fa")
        self.result_display.place(relx=0.5, rely=0.5, anchor="center")

        self.res_label = tk.Label(self.status_card, text="Оберіть файл для початку",
                                  font=("Segoe UI", 10), bg="#f8f9fa", fg="#7f8c8d")
        self.res_label.place(relx=0.5, rely=0.5, anchor="center")

        self.path_label = tk.Label(self.main_frame, text="", font=("Segoe UI", 8),
                                   bg="#ffffff", fg="#b2bec3", wraplength=700)
        self.path_label.pack(pady=(0, 15))

        btn_frame = tk.Frame(self.main_frame, bg="#ffffff")
        btn_frame.pack(fill=tk.X)

        self.btn_select = tk.Button(btn_frame, text="ВИБРАТИ ФАЙЛ", command=self.select_file,
                                    bg="#000000", fg="white", font=("Segoe UI", 10, "bold"),
                                    relief="flat", width=20, pady=12, cursor="hand2")
        self.btn_select.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.btn_run = tk.Button(btn_frame, text="АНАЛІЗ", command=self.process,
                                 bg="#666666", fg="white", font=("Segoe UI", 10, "bold"),
                                 relief="flat", width=20, pady=12, cursor="",
                                 state=tk.DISABLED, disabledforeground="#f0f0f0")
        self.btn_run.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        self.btn_save = tk.Button(btn_frame, text="ЗБЕРЕГТИ", command=self.save_result,
                                  bg="#666666", fg="white", font=("Segoe UI", 10, "bold"),
                                  relief="flat", width=20, pady=12, cursor="",
                                  state=tk.DISABLED, disabledforeground="#f0f0f0")
        self.btn_save.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Зображення", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.path = file_path
            self.original_filename = os.path.splitext(os.path.basename(self.path))[0]
            self.path_label.config(text=f"Файл: {self.path}")
            img = cv2.imread(self.path)
            if img is not None:
                self.res_label.lower()
                self.show_image_in_tk(img)
            self.btn_run.config(state=tk.NORMAL, bg="#000000", cursor="hand2")
            self.btn_save.config(state=tk.DISABLED, bg="#666666", cursor="")
            self.header.config(text="Аналіз зображень")

    def process(self):
        if not self.path: return
        self.btn_run.config(state=tk.DISABLED, text="...", bg="#666666")
        self.root.update()

        try:
            img = cv2.imread(self.path)
            h, w = img.shape[:2]

            # Покращуємо контрастність
            img_enhanced = self.apply_clahe(img)

            # Запуск моделі на всьому зображенні одразу
            # imgsz=1024 допомагає бачити дрібні деталі на великих фото
            res = model.predict(source=img_enhanced, imgsz=1024, conf=self.CONF_DEFAULT, iou=0.6)

            color_layer = np.zeros((h, w, 3), dtype=np.uint8)
            count = 0

            if res[0].masks is not None:
                # Отримуємо маски та масштабуємо їх до розміру оригіналу
                masks = res[0].masks.data.cpu().numpy()

                for m in masks:
                    # Resize маски до розмірів вхідного фото
                    mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    # Фільтрація за площею
                    area = np.sum(mask_binary)
                    if area < self.AREA_DEFAULT:
                        continue

                    count += 1
                    # Генерація випадкового кольору для кожного каменя
                    color = np.random.randint(40, 255, (3,)).tolist()
                    color_layer[mask_binary == 1] = color

            # Накладання масок на оригінальне фото
            self.last_result = cv2.addWeighted(img, 0.7, color_layer, 0.3, 0)

            self.show_image_in_tk(self.last_result)
            self.header.config(text=f"Знайдено: {count} шт.")
            self.btn_save.config(state=tk.NORMAL, bg="#000000", cursor="hand2")

        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка: {str(e)}")
        finally:
            self.btn_run.config(state=tk.NORMAL, text="АНАЛІЗ", bg="#000000", cursor="hand2")

    def save_result(self):
        if self.last_result is not None:
            save_path = filedialog.asksaveasfilename(
                initialfile=f"{self.original_filename}_result.jpg",
                defaultextension=".jpg"
            )
            if save_path:
                cv2.imwrite(save_path, self.last_result)
                messagebox.showinfo("Успіх", "Збережено!")

    def show_image_in_tk(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((600, 400), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.result_display.config(image=self.tk_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = StoneCounter(root)
    root.mainloop()