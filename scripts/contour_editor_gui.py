import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# ---------- helpers ----------
def bgr_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def nppts_to_cv(pts):
    # list[(x,y)] -> cv2 contour (-1,1,2)
    if len(pts) == 0:
        return np.zeros((0,1,2), dtype=np.int32)
    arr = np.array(pts, dtype=np.int32).reshape(-1,1,2)
    return arr

def cv_to_nppts(contour):
    # cv2 contour -> list[(x,y)]
    return [(int(p[0][0]), int(p[0][1])) for p in contour]

# ---------- main editor ----------
class ContourEditorGUI:
    """
    Windows-style GUI editor embedded in Tkinter.
    - Draw new independent contours (New Boundary)
    - Edit vertices (Edit mode)
    - Delete vertex (Right-click)
    - Box-delete vertices (Right-click + drag)
    - Join two contours (J: select A then B)
    - Zoom (+/-), Pan (hold SPACE and drag or middle mouse)
    - Save / Cancel / Reset
    """

    def __init__(self, image_bgr, initial_contours=None):
        # data
        self.base_img_bgr = image_bgr.copy()
        self.h, self.w = self.base_img_bgr.shape[:2]
        self.contours = [cv_to_nppts(c) for c in (initial_contours or [])]  # list[list[(x,y)]]
        self.current_poly = []   # while drawing
        self.mode = "view"       # view | draw | edit | join
        self.join_first_idx = None

        # view transform
        self.scale = 1.0
        self.min_scale = 0.25
        self.max_scale = 5.0
        self.offset = [0, 0]    # pan offset in screen pixels
        self.panning = False
        self.last_mouse = None

        # edit state
        self.selected = None     # (contour_idx, vertex_idx)
        self.hover = None

        # box selection (right-drag)
        self.box_selecting = False
        self.box_start_img = None      # (x, y) in image coords
        self.box_current_img = None    # (x, y) in image coords
        self.box_start_screen = None   # (x, y) in screen coords

        # tk setup
        self.root = tk.Tk()
        self.root.title("Contour Editor - Review & Adjust Boundaries")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # top toolbar
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.btn_new = ttk.Button(bar, text="New Boundary", command=self.toggle_draw)
        self.btn_edit = ttk.Button(bar, text="Edit (Move Vertices)", command=self.toggle_edit)
        self.btn_join = ttk.Button(bar, text="Join (J)", command=self.toggle_join)
        self.btn_reset = ttk.Button(bar, text="Reset", command=self.reset_all)
        self.btn_zoom_in = ttk.Button(bar, text="Zoom +", command=lambda: self.zoom(1.2))
        self.btn_zoom_out = ttk.Button(bar, text="Zoom -", command=lambda: self.zoom(1/1.2))
        self.btn_save = ttk.Button(bar, text="Save", command=self.on_save)
        self.btn_cancel = ttk.Button(bar, text="Cancel", command=self.on_cancel)

        for w in (self.btn_new, self.btn_edit, self.btn_join, self.btn_reset,
                  self.btn_zoom_in, self.btn_zoom_out, self.btn_save, self.btn_cancel):
            w.pack(side=tk.LEFT, padx=5)

        # canvas
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=1, highlightbackground="#aaa")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # bind events
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)

        # right button: click = delete single; drag = box-select delete
        self.canvas.bind("<Button-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_up)

        self.root.bind("<Key-j>", lambda e: self.toggle_join())
        self.root.bind("<Key-Escape>", lambda e: self.cancel_current())
        self.root.bind("<Key-s>", lambda e: self.on_save())
        self.root.bind("<Key-c>", lambda e: self.clear_current())
        self.root.bind("<Key-plus>", lambda e: self.zoom(1.2))
        self.root.bind("<Key-minus>", lambda e: self.zoom(1/1.2))
        self.root.bind("<Key-space>", self.space_down)
        self.root.bind("<KeyRelease-space>", self.space_up)

        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Button-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)

        # initial render
        self.tk_img = None
        self.result = None
        self.render()

    # ---------- mode toggles ----------
    def toggle_draw(self):
        if self.mode == "draw":
            self.mode = "view"
            self.btn_new.state(["!pressed"])
        else:
            self.mode = "draw"
            self.btn_new.state(["pressed"])
            self.btn_edit.state(["!pressed"])
            self.btn_join.state(["!pressed"])

    def toggle_edit(self):
        if self.mode == "edit":
            self.mode = "view"
            self.btn_edit.state(["!pressed"])
        else:
            self.mode = "edit"
            self.btn_edit.state(["pressed"])
            self.btn_new.state(["!pressed"])
            self.btn_join.state(["!pressed"])

    def toggle_join(self):
        if self.mode == "join":
            self.mode = "view"
            self.join_first_idx = None
            self.btn_join.state(["!pressed"])
        else:
            self.mode = "join"
            self.join_first_idx = None
            self.btn_join.state(["pressed"])
            self.btn_new.state(["!pressed"])
            self.btn_edit.state(["!pressed"])

    # ---------- pan/zoom ----------
    def screen_to_image(self, sx, sy):
        x = (sx - self.offset[0]) / self.scale
        y = (sy - self.offset[1]) / self.scale
        return int(round(x)), int(round(y))

    def zoom(self, factor):
        new_scale = np.clip(self.scale * factor, self.min_scale, self.max_scale)
        self.scale = float(new_scale)
        self.render()

    def space_down(self, _):
        self.panning = True

    def space_up(self, _):
        self.panning = False
        self.last_mouse = None

    def on_middle_down(self, e):
        self.panning = True
        self.last_mouse = (e.x, e.y)

    def on_middle_drag(self, e):
        if self.panning and self.last_mouse:
            dx = e.x - self.last_mouse[0]
            dy = e.y - self.last_mouse[1]
            self.offset[0] += dx
            self.offset[1] += dy
            self.last_mouse = (e.x, e.y)
            self.render()

    def on_middle_up(self, _):
        self.panning = False
        self.last_mouse = None

    # ---------- left-button drawing & editing ----------
    def on_left_down(self, e):
        ix, iy = self.screen_to_image(e.x, e.y)
        if self.panning:
            self.last_mouse = (e.x, e.y)
            return

        if self.mode == "draw":
            # start or continue current polygon
            if not self.current_poly:
                self.current_poly = [(ix, iy)]
            else:
                # double-click closes polygon
                if len(self.current_poly) >= 3 and self.dist(self.current_poly[0], (ix, iy)) < 10:
                    self.finish_current_poly()
                else:
                    self.current_poly.append((ix, iy))
            self.render()
        elif self.mode == "edit":
            # pick nearest vertex to drag
            self.selected = self.find_nearest_vertex((ix, iy), thresh=10)
        elif self.mode == "join":
            idx = self.find_contour_hit((ix, iy), thresh=10)
            if idx is not None:
                if self.join_first_idx is None:
                    self.join_first_idx = idx
                else:
                    if idx != self.join_first_idx:
                        self.join_contours(self.join_first_idx, idx)
                    self.join_first_idx = None
                self.render()

    def on_left_drag(self, e):
        if self.panning:
            if self.last_mouse:
                dx = e.x - self.last_mouse[0]
                dy = e.y - self.last_mouse[1]
                self.offset[0] += dx
                self.offset[1] += dy
                self.last_mouse = (e.x, e.y)
                self.render()
            return

        ix, iy = self.screen_to_image(e.x, e.y)
        if self.mode == "edit" and self.selected:
            ci, vi = self.selected
            if 0 <= ci < len(self.contours) and 0 <= vi < len(self.contours[ci]):
                self.contours[ci][vi] = (ix, iy)
                self.render()

    def on_left_up(self, _):
        self.selected = None

    # ---------- right-button delete / box-select delete ----------
    def delete_single_vertex_at(self, ix, iy, thresh=10):
        """Delete the nearest vertex around (ix, iy) in image coords."""
        hit = self.find_nearest_vertex((ix, iy), thresh=thresh)
        if hit:
            ci, vi = hit
            if len(self.contours[ci]) > 3:
                del self.contours[ci][vi]
            else:
                # too small -> remove contour entirely
                del self.contours[ci]
            self.render()

    def delete_points_in_box(self, x_min, y_min, x_max, y_max):
        """Delete all vertices inside the axis-aligned box in image coords."""
        new_contours = []
        for poly in self.contours:
            kept = [
                (x, y) for (x, y) in poly
                if not (x_min <= x <= x_max and y_min <= y <= y_max)
            ]
            if len(kept) >= 3:
                new_contours.append(kept)
        self.contours = new_contours
        self.render()

    def on_right_down(self, e):
        """Start single-click or box selection."""
        ix, iy = self.screen_to_image(e.x, e.y)
        self.box_selecting = True
        self.box_start_img = (ix, iy)
        self.box_current_img = (ix, iy)
        self.box_start_screen = (e.x, e.y)

    def on_right_drag(self, e):
        """Update box selection while dragging."""
        if not self.box_selecting:
            return
        ix, iy = self.screen_to_image(e.x, e.y)
        self.box_current_img = (ix, iy)
        self.render()  # will draw the green rectangle

    def on_right_up(self, e):
        """End click/drag: either delete single vertex or box-delete."""
        if not self.box_selecting:
            return

        dx = e.x - self.box_start_screen[0]
        dy = e.y - self.box_start_screen[1]
        move_dist_sq = dx * dx + dy * dy

        start_img = self.box_start_img
        end_img = self.box_current_img

        # reset selection state BEFORE deleting so rectangle disappears
        self.box_selecting = False
        self.box_start_img = None
        self.box_current_img = None
        self.box_start_screen = None

        # tiny movement => treat as simple right-click delete
        click_thresh_sq = 5 * 5  # 5 px radius
        if move_dist_sq <= click_thresh_sq:
            if start_img is not None:
                self.delete_single_vertex_at(*start_img)
            return

        # otherwise, do a box delete
        if start_img is not None and end_img is not None:
            x1, y1 = start_img
            x2, y2 = end_img
            x_min, x_max = sorted((x1, x2))
            y_min, y_max = sorted((y1, y2))
            self.delete_points_in_box(x_min, y_min, x_max, y_max)

    # ---------- mouse move ----------
    def on_motion(self, e):
        ix, iy = self.screen_to_image(e.x, e.y)
        self.hover = self.find_nearest_vertex((ix, iy), thresh=8)
        # could use this to highlight hover vertex later

    def finish_current_poly(self):
        if len(self.current_poly) >= 3:
            self.contours.append(self.current_poly.copy())
        self.current_poly = []
        self.render()

    # ---------- joins ----------
    def join_contours(self, a_idx, b_idx):
        # simple concat: connect end of A to start of B
        if 0 <= a_idx < len(self.contours) and 0 <= b_idx < len(self.contours):
            A = self.contours[a_idx]
            B = self.contours[b_idx]
            # choose endpoints nearest to link
            Ax0, Ay0 = A[0]
            Ax1, Ay1 = A[-1]
            Bx0, By0 = B[0]
            Bx1, By1 = B[-1]

            choices = [
                (self.dist((Ax1, Ay1), (Bx0, By0)), A + B),             # A->B
                (self.dist((Ax1, Ay1), (Bx1, By1)), A + B[::-1]),       # A->revB
                (self.dist((Ax0, Ay0), (Bx0, By0)), A[::-1] + B),       # revA->B
                (self.dist((Ax0, Ay0), (Bx1, By1)), A[::-1] + B[::-1])  # revA->revB
            ]
            choices.sort(key=lambda t: t[0])
            merged = choices[0][1]
            # replace a_idx with merged, remove b_idx
            self.contours[a_idx] = merged
            if b_idx > a_idx:
                del self.contours[b_idx]
            else:
                del self.contours[a_idx+1]

    # ---------- finders ----------
    def find_nearest_vertex(self, p, thresh=10):
        best = None
        best_d = 1e9
        for ci, poly in enumerate(self.contours):
            for vi, q in enumerate(poly):
                d = self.dist(p, q)
                if d < best_d and d <= thresh:
                    best = (ci, vi)
                    best_d = d
        return best

    def find_contour_hit(self, p, thresh=10):
        # nearest vertex implies that contour is selected
        hit = self.find_nearest_vertex(p, thresh)
        return hit[0] if hit else None

    @staticmethod
    def dist(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    # ---------- UI commands ----------
    def reset_all(self):
        if messagebox.askyesno("Reset", "Clear current drawing (not saved contours)?"):
            self.current_poly = []
            self.render()

    def cancel_current(self):
        self.current_poly = []
        self.mode = "view"
        self.btn_new.state(["!pressed"])
        self.btn_edit.state(["!pressed"])
        self.btn_join.state(["!pressed"])
        self.render()

    def clear_current(self):
        # keyboard 'c' clears current unfinished poly
        self.current_poly = []
        self.render()

    def on_save(self):
        # finalize current poly if drawing
        if self.current_poly and len(self.current_poly) >= 3:
            self.finish_current_poly()
        self.result = [nppts_to_cv(poly) for poly in self.contours]
        self.root.destroy()

    def on_cancel(self):
        if messagebox.askyesno("Cancel", "Discard edits and close?"):
            self.result = None
            self.root.destroy()

    # ---------- rendering ----------
    def render(self):
        # compose display image with contours
        disp = self.base_img_bgr.copy()

        # draw existing contours
        for poly in self.contours:
            if len(poly) >= 2:
                pts = np.array(poly, dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(disp, [pts], isClosed=True, color=(0,0,255), thickness=2)
            for (x,y) in poly:
                cv2.circle(disp, (x,y), 3, (0,255,255), -1)

        # draw current polygon (green)
        if len(self.current_poly) >= 2:
            pts = np.array(self.current_poly, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(disp, [pts], isClosed=False, color=(0,255,0), thickness=2)
            for (x,y) in self.current_poly:
                cv2.circle(disp, (x,y), 3, (0,200,0), -1)

        # apply pan+zoom
        pil = bgr_to_pil(disp)
        W = int(self.w * self.scale)
        H = int(self.h * self.scale)
        pil = pil.resize((W, H), resample=Image.BILINEAR)

        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        # draw with offset
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.tk_img)

        # helpful status bar text
        mode_text = f"Mode: {self.mode.upper()}  |  Zoom: {self.scale:.2f}x  |  Contours: {len(self.contours)}"
        self.canvas.create_text(10, 10, anchor="nw", text=mode_text, fill="white", font=("Segoe UI", 10, "bold"))

        # draw box-selection rectangle if active
        if self.box_selecting and self.box_start_img and self.box_current_img:
            x1, y1 = self.box_start_img
            x2, y2 = self.box_current_img

            # convert image coords -> screen/canvas coords
            sx1 = x1 * self.scale + self.offset[0]
            sy1 = y1 * self.scale + self.offset[1]
            sx2 = x2 * self.scale + self.offset[0]
            sy2 = y2 * self.scale + self.offset[1]

            self.canvas.create_rectangle(sx1, sy1, sx2, sy2,
                                         outline="lime", width=1)

    # ---------- public API ----------
    def start(self):
        self.root.mainloop()
        return self.result
