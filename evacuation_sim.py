#!/usr/bin/env python3
"""
MoSi - Particle Swarm Optimization-Based Simulation of Emergency Evacuation Routes
under varying layouts, starting locations and exit configurations.

Pure Python + tkinter (standard library). No external dependencies.
Run:  python evacuation_sim.py
"""

import math
import random
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

INF = float("inf")

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
class Vec:
    __slots__ = ("x", "y")
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
    def __add__(self, o):
        return Vec(self.x + o.x, self.y + o.y)
    def __sub__(self, o):
        return Vec(self.x - o.x, self.y - o.y)
    def __mul__(self, k):
        return Vec(self.x * k, self.y * k)
    def __rmul__(self, k):
        return self.__mul__(k)
    def dist(self, o):
        return math.hypot(self.x - o.x, self.y - o.y)
    def length(self):
        return math.hypot(self.x, self.y)
    def normalized(self):
        L = self.length()
        return Vec(self.x / L, self.y / L) if L > 0 else Vec(0.0, 0.0)
    def copy(self):
        return Vec(self.x, self.y)
    def clamp(self, lo, hi):
        return Vec(min(max(self.x, lo), hi), min(max(self.y, lo), hi))

class Rect:
    __slots__ = ("x", "y", "w", "h")
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
    @property
    def left(self):
        return self.x
    @property
    def right(self):
        return self.x + self.w
    @property
    def top(self):
        return self.y
    @property
    def bottom(self):
        return self.y + self.h
    @property
    def center(self):
        return Vec(self.x + self.w / 2.0, self.y + self.h / 2.0)
    def contains_point(self, p):
        return self.left <= p.x <= self.right and self.top <= p.y <= self.bottom
    def dist_to_point(self, p):
        cx = min(max(p.x, self.left), self.right)
        cy = min(max(p.y, self.top), self.bottom)
        return Vec(cx, cy).dist(p)
    def copy(self):
        return Rect(self.x, self.y, self.w, self.h)
    def inflated(self, d):
        return Rect(self.x - d, self.y - d, self.w + 2 * d, self.h + 2 * d)

def push_out_of_walls(pos, walls, radius, iters=3):
    """Nudge a point out of any wall it overlaps (inflated by radius)."""
    for _ in range(iters):
        moved = False
        for w in walls:
            r = w.inflated(radius)
            if r.contains_point(pos):
                cx, cy = r.center.x, r.center.y
                ox = min(pos.x - r.left, r.right - pos.x)
                oy = min(pos.y - r.top, r.bottom - pos.y)
                if ox <= oy:
                    if pos.x < cx:
                        pos.x = r.left - 0.01
                    else:
                        pos.x = r.right + 0.01
                else:
                    if pos.y < cy:
                        pos.y = r.top - 0.01
                    else:
                        pos.y = r.bottom + 0.01
                moved = True
        if not moved:
            break
    return pos

# ---------------------------------------------------------------------------
# Particle & PSO
# ---------------------------------------------------------------------------
class Particle:
    def __init__(self, x, y, radius=7.0):
        self.pos = Vec(x, y)
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(2.0, 8.0)
        self.vel = Vec(math.cos(ang) * spd, math.sin(ang) * spd)
        self.pbest_pos = Vec(x, y)
        self.pbest_cost = INF
        self.radius = float(radius)
        self.target_kind = "nearest"   # nearest | exit | safe
        self.target_obj = None         # Rect target (exit or safe zone)
        self.evacuated = False
    def target_rect(self, exits, safes):
        if self.target_obj is not None:
            return self.target_obj
        if not exits:
            return None
        best = min(exits, key=lambda e: self.pos.dist(e.center))
        return best
    def target_pos(self, exits, safes):
        t = self.target_rect(exits, safes)
        return t.center if t else self.pos.copy()
    def group_key(self, exits, safes):
        t = self.target_rect(exits, safes)
        if t is None:
            return ("none", 0)
        if t in exits:
            return ("exit", exits.index(t))
        if t in safes:
            return ("safe", safes.index(t))
        return ("none", 0)
    def cost(self, walls, exits, safes):
        t = self.target_pos(exits, safes)
        c = self.pos.dist(t)
        r = self.radius
        for w in walls:
            d = w.dist_to_point(self.pos)
            if d < r:
                c += 1000.0 + (r - d) * 50.0
        return c

class PSOEngine:
    def __init__(self, walls, exits, safes, particles,
                 w=0.8, c1=1.5, c2=1.5, max_vel=11.0):
        self.walls = walls
        self.exits = exits
        self.safes = safes
        self.particles = particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_vel = max_vel
        self.iteration = 0
    def step(self, bounds=None):
        # Lock 'nearest' particles to a specific exit at the start of a run.
        for p in self.particles:
            if p.evacuated:
                continue
            if p.target_kind == "nearest" and p.target_obj is None:
                p.target_obj = p.target_rect(self.exits, self.safes)
        # Group particles by their target so each group converges to its own exit/safe.
        groups = {}
        for p in self.particles:
            if p.evacuated:
                continue
            groups.setdefault(p.group_key(self.exits, self.safes), []).append(p)
        gbest = {}
        for key, plist in groups.items():
            best = min(plist, key=lambda p: p.pbest_cost)
            gbest[key] = best.pbest_pos.copy()
        reached = 0
        for p in self.particles:
            if p.evacuated:
                reached += 1
                continue
            key = p.group_key(self.exits, self.safes)
            g = gbest.get(key, p.pos.copy())
            r1, r2 = random.random(), random.random()
            p.vel = p.vel * self.w \
                + (p.pbest_pos - p.pos) * (self.c1 * r1) \
                + (g - p.pos) * (self.c2 * r2)
            # stochastic exploration jitter keeps the swarm from
            # prematurely converging to a mediocre group-best
            p.vel = p.vel + Vec(random.uniform(-1.5, 1.5),
                                random.uniform(-1.5, 1.5))
            L = p.vel.length()
            if L > self.max_vel:
                p.vel = p.vel.normalized() * self.max_vel
            newpos = p.pos + p.vel
            newpos = push_out_of_walls(newpos, self.walls, p.radius)
            if bounds is not None:
                newpos = newpos.clamp(p.radius, bounds.x - p.radius)
                newpos.y = min(max(newpos.y, p.radius), bounds.y - p.radius)
            p.pos = newpos
            c = p.cost(self.walls, self.exits, self.safes)
            if c < p.pbest_cost:
                p.pbest_cost = c
                p.pbest_pos = p.pos.copy()
            if p.pos.dist(p.target_pos(self.exits, self.safes)) < 14.0:
                p.evacuated = True
                reached += 1
        self.iteration += 1
        return reached


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
CANVAS_W = 1100
CANVAS_H = 700
MIN_RECT = 12.0

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MoSi - PSO Emergency Evacuation Simulation")
        self.geometry("1400x800")

        self.walls = []
        self.exits = []
        self.safes = []
        self.particles = []

        self.mode = "EDIT"
        self.tool = "select"
        self.selected = None
        self.pending_target = None
        self.drag = None
        self.engine = None
        self.sim_running = False
        self.sim_job = None

        self.pso_w = tk.DoubleVar(value=0.7)
        self.pso_c1 = tk.DoubleVar(value=1.5)
        self.pso_c2 = tk.DoubleVar(value=1.5)
        self.pso_maxvel = tk.DoubleVar(value=11.0)
        self.pso_iters = tk.IntVar(value=1)

        self.show_vel = tk.BooleanVar(value=True)
        self.show_pbest = tk.BooleanVar(value=True)
        self.show_gbest = tk.BooleanVar(value=True)

        self.build_ui()
        self.redraw()
        self.status("Edit mode: pick a tool on the left. Click-drag to draw walls/exits/safe zones; click to place particles.")

    def build_ui(self):
        left = ttk.Frame(self, width=230)
        left.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        left.grid_propagate(False)

        ttk.Label(left, text="MODE", font=("Segoe UI", 10, "bold")).pack(pady=(6, 2))
        self.btn_mode_edit = ttk.Button(left, text="Edit Map", width=18, command=self.set_edit_mode)
        self.btn_mode_edit.pack(pady=2)
        self.btn_mode_sim = ttk.Button(left, text="Simulate", width=18, command=self.set_sim_mode)
        self.btn_mode_sim.pack(pady=2)

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="EDIT TOOL", font=("Segoe UI", 10, "bold")).pack(pady=(2, 2))
        self.tool_btns = {}
        for txt, val in [("Select / Move", "select"),
                         ("Draw Wall", "wall"),
                         ("Draw Exit", "exit"),
                         ("Draw Safe Zone", "safe"),
                         ("Place Particle", "particle"),
                         ("Delete", "delete"),
                         ("Set Target", "target")]:
            b = ttk.Button(left, text=txt, width=18, command=lambda v=val: self.set_tool(v))
            b.pack(pady=2)
            self.tool_btns[val] = b

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="SIMULATION", font=("Segoe UI", 10, "bold")).pack(pady=(2, 2))
        self.btn_run = ttk.Button(left, text="Run", width=18, command=self.toggle_run)
        self.btn_run.pack(pady=2)
        ttk.Button(left, text="Step 1 Iter", width=18, command=self.step_once).pack(pady=2)
        ttk.Button(left, text="Reset Sim", width=18, command=self.reset_sim).pack(pady=2)

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Checkbutton(left, text="Show velocity arrows", variable=self.show_vel,
                        command=self.redraw).pack(anchor="w", pady=1)
        ttk.Checkbutton(left, text="Show personal best", variable=self.show_pbest,
                        command=self.redraw).pack(anchor="w", pady=1)
        ttk.Checkbutton(left, text="Show group best", variable=self.show_gbest,
                        command=self.redraw).pack(anchor="w", pady=1)

        self.canvas = tk.Canvas(self, width=CANVAS_W, height=CANVAS_H, bg="#f7f7f7",
                                highlightthickness=1, highlightbackground="#bbb")
        self.canvas.grid(row=0, column=1, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_motion)

        right = ttk.Frame(self, width=300)
        right.grid(row=0, column=2, sticky="ns", padx=5, pady=5)
        right.grid_propagate(False)
        self.build_pso_panel(right)
        self.build_selection_panel(right)
        self.build_map_panel(right)
        self.build_stats_panel(right)

        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self, textvariable=self.status_var, relief="sunken",
                           anchor="w", padding=4)
        status.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.refresh_tool_buttons()

    def build_pso_panel(self, parent):
        p = ttk.LabelFrame(parent, text="PSO Parameters", padding=8)
        p.pack(fill="x", pady=4)

        def scale_row(label, var, frm, to):
            ttk.Label(p, text=label).pack(anchor="w")
            ttk.Scale(p, from_=frm, to=to, variable=var, orient="horizontal").pack(fill="x", pady=(0, 2))
            ttk.Entry(p, textvariable=var, width=8).pack(anchor="e", pady=(0, 6))

        scale_row("Inertia (w)", self.pso_w, 0.0, 2.0)
        scale_row("Cognitive (c1)", self.pso_c1, 0.0, 4.0)
        scale_row("Social (c2)", self.pso_c2, 0.0, 4.0)
        scale_row("Max velocity", self.pso_maxvel, 1.0, 20.0)
        scale_row("Iters per frame", self.pso_iters, 1, 20)

        ttk.Label(p, text="Add particles:").pack(anchor="w")
        self.add_count_var = tk.IntVar(value=10)
        row = ttk.Frame(p)
        row.pack(fill="x", pady=(0, 6))
        ttk.Entry(row, textvariable=self.add_count_var, width=6).pack(side="left")
        ttk.Button(row, text="Add", width=8, command=self.add_particles).pack(side="left", padx=5)
        ttk.Button(p, text="Apply PSO Params", command=self.apply_pso).pack(fill="x", pady=(4, 0))

    def build_selection_panel(self, parent):
        p = ttk.LabelFrame(parent, text="Selected Object", padding=8)
        p.pack(fill="x", pady=4)

        self.sel_kind_var = tk.StringVar(value="-")
        ttk.Label(p, textvariable=self.sel_kind_var, font=("Segoe UI", 9, "bold")).pack(anchor="w")

        ttk.Label(p, text="Target kind:").pack(anchor="w", pady=(8, 0))
        self.sel_target_kind = tk.StringVar(value="nearest")
        ttk.Combobox(p, textvariable=self.sel_target_kind,
                     values=["nearest", "exit", "safe"], state="readonly", width=14).pack(anchor="w")
        ttk.Button(p, text="Apply Target to Selected", command=self.apply_target).pack(fill="x", pady=(4, 2))

        ttk.Label(p, text="Width:").pack(anchor="w", pady=(8, 0))
        self.sel_w_var = tk.DoubleVar(value=0)
        self.sel_w_entry = ttk.Entry(p, textvariable=self.sel_w_var, width=10)
        self.sel_w_entry.pack(anchor="w")
        ttk.Label(p, text="Height:").pack(anchor="w", pady=(4, 0))
        self.sel_h_var = tk.DoubleVar(value=0)
        self.sel_h_entry = ttk.Entry(p, textvariable=self.sel_h_var, width=10)
        self.sel_h_entry.pack(anchor="w", pady=(0, 4))
        ttk.Button(p, text="Apply Size", command=self.apply_size).pack(fill="x")
        ttk.Button(p, text="Delete Selected", command=self.delete_selected).pack(fill="x", pady=(4, 0))

        self.sel_w_entry.bind("<FocusOut>", lambda e: self.apply_size())
        self.sel_h_entry.bind("<FocusOut>", lambda e: self.apply_size())

    def build_map_panel(self, parent):
        p = ttk.LabelFrame(parent, text="Map", padding=8)
        p.pack(fill="x", pady=4)
        ttk.Button(p, text="Save Map", command=self.save_map).pack(fill="x", pady=2)
        ttk.Button(p, text="Load Map", command=self.load_map).pack(fill="x", pady=2)
        ttk.Button(p, text="Clear Map", command=self.clear_map).pack(fill="x", pady=2)
        ttk.Button(p, text="Demo Layout", command=self.demo_layout).pack(fill="x", pady=(6, 0))

    def build_stats_panel(self, parent):
        p = ttk.LabelFrame(parent, text="Statistics", padding=8)
        p.pack(fill="x", pady=4)
        self.stats_iter = tk.StringVar(value="Iter: 0")
        self.stats_particles = tk.StringVar(value="Particles: 0")
        self.stats_evac = tk.StringVar(value="Evacuated: 0")
        for var in (self.stats_iter, self.stats_particles, self.stats_evac):
            ttk.Label(p, textvariable=var).pack(anchor="w")

    # ------------------------------------------------------------------ helpers --
    def status(self, msg):
        self.status_var.set(msg)

    def refresh_tool_buttons(self):
        for val, b in self.tool_btns.items():
            b.state(["!pressed" if val != self.tool else "pressed"])

    def set_edit_mode(self):
        self.mode = "EDIT"
        self.sim_running = False
        self.btn_run.config(text="Run")
        self.status("Edit mode active.")
        self.refresh_tool_buttons()
        self.redraw()

    def set_sim_mode(self):
        self.mode = "SIM"
        if not self.particles:
            self.status("Sim mode: no particles yet. Add particles in Edit mode, then Run.")
        else:
            self.status("Sim mode: click Run to start PSO.")
        self.refresh_tool_buttons()
        self.redraw()

    def set_tool(self, tool):
        self.tool = tool
        self.pending_target = None
        self.selected = None
        self.status("Tool: " + tool + (" (click a particle, then an exit/safe zone)" if tool == "target" else ""))
        self.refresh_tool_buttons()
        self.update_selection_panel()
        self.redraw()

    def apply_pso(self):
        try:
            w = float(self.pso_w.get()); c1 = float(self.pso_c1.get())
            c2 = float(self.pso_c2.get()); mv = float(self.pso_maxvel.get())
            it = int(self.pso_iters.get())
        except Exception:
            messagebox.showerror("Invalid PSO parameters", "Check the numeric PSO values.")
            return
        if self.engine is None:
            self.engine = PSOEngine(self.walls, self.exits, self.safes, self.particles)
        self.engine.w = w; self.engine.c1 = c1; self.engine.c2 = c2
        self.engine.max_vel = mv; self.engine.iteration = 0
        self.status("PSO parameters applied.")

    def add_particles(self):
        n = max(1, int(self.add_count_var.get()))
        placed = 0
        tries = 0
        while placed < n and tries < n * 40:
            tries += 1
            x = random.uniform(30, CANVAS_W - 30)
            y = random.uniform(30, CANVAS_H - 30)
            p = Particle(x, y)
            if all(w.dist_to_point(p.pos) >= p.radius + 2 for w in self.walls):
                self.particles.append(p)
                placed += 1
        self.stats_particles.set(f"Particles: {len(self.particles)}")
        self.status(f"Added {placed} particle(s).")
        self.redraw()

    def apply_target(self):
        if self.selected is None or not isinstance(self.selected, Particle):
            messagebox.showinfo("No particle selected", "Select a particle first (Select tool).")
            return
        kind = self.sel_target_kind.get()
        self.selected.target_kind = kind
        if kind == "nearest":
            self.selected.target_obj = None
        elif kind == "exit":
            self.selected.target_obj = self.exits[0] if self.exits else None
        elif kind == "safe":
            self.selected.target_obj = self.safes[0] if self.safes else None
        if self.selected.target_obj is None and kind != "nearest":
            messagebox.showwarning("No target", f"No {kind} zones exist yet. Draw one first.")
        self.status(f"Target for selected particle -> {kind}.")
        self.redraw()

    def apply_size(self):
        if self.selected is None or not hasattr(self.selected, "w"):
            return
        try:
            w = max(MIN_RECT, float(self.sel_w_var.get()))
            h = max(MIN_RECT, float(self.sel_h_var.get()))
        except Exception:
            return
        self.selected.w = w
        self.selected.h = h
        self.redraw()

    def delete_selected(self):
        if self.selected is None:
            return
        lst = {"wall": self.walls, "exit": self.exits, "safe": self.safes,
               "particle": self.particles}.get(self.selected.kind)
        if lst is not None and self.selected in lst:
            lst.remove(self.selected)
        if self.engine and self.selected in self.engine.particles:
            self.engine.particles.remove(self.selected)
        self.selected = None
        self.stats_particles.set(f"Particles: {len(self.particles)}")
        self.update_selection_panel()
        self.status("Deleted selected object.")
        self.redraw()

    def clear_map(self):
        self.walls.clear(); self.exits.clear(); self.safes.clear(); self.particles.clear()
        self.selected = None
        self.engine = None
        self.sim_running = False
        self.btn_run.config(text="Run")
        self.stats_iter.set("Iter: 0")
        self.stats_evac.set("Evacuated: 0")
        self.stats_particles.set("Particles: 0")
        self.status("Map cleared.")
        self.redraw()

    def demo_layout(self):
        self.clear_map()
        # Open room with perimeter exits and a few non-blocking inner obstacles.
        self.walls += [
            Rect(430, 200, 140, 110),   # central pillar
            Rect(620, 360, 110, 60),    # inner obstacle
            Rect(260, 410, 100, 45),    # inner obstacle
        ]
        self.exits += [Rect(20, 290, 30, 120), Rect(1050, 290, 30, 120)]
        self.safes += [Rect(470, 640, 90, 50), Rect(930, 130, 50, 70)]
        for _ in range(16):
            self.add_particles()
        self.add_count_var.set(0)
        self.stats_particles.set(f"Particles: {len(self.particles)}")
        self.status("Demo layout loaded. Switch to Sim mode and Run (PSO steers particles to exits/safe zones).")

    def save_map(self):
        data = {
            "walls": [[r.x, r.y, r.w, r.h] for r in self.walls],
            "exits": [[r.x, r.y, r.w, r.h] for r in self.exits],
            "safes": [[r.x, r.y, r.w, r.h] for r in self.safes],
            "particles": [[p.pos.x, p.pos.y] for p in self.particles],
        }
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
        self.status(f"Map saved to {path}.")

    def load_map(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.clear_map()
        for w in data.get("walls", []):
            self.walls.append(Rect(*w))
        for e in data.get("exits", []):
            self.exits.append(Rect(*e))
        for s in data.get("safes", []):
            self.safes.append(Rect(*s))
        for px, py in data.get("particles", []):
            self.particles.append(Particle(px, py))
        self.stats_particles.set(f"Particles: {len(self.particles)}")
        self.engine = None
        self.status(f"Map loaded from {path}.")
        self.redraw()

    # ----------------------------------------------- selection & hit testing --
    def update_selection_panel(self):
        obj = self.selected
        if obj is None:
            self.sel_kind_var.set("-")
            self.sel_target_kind.set("nearest")
            self.sel_w_var.set(0)
            self.sel_h_var.set(0)
            self.sel_w_entry.state(["disabled"])
            self.sel_h_entry.state(["disabled"])
            return
        kind = getattr(obj, "kind", "?")
        self.sel_kind_var.set(f"{kind.upper()}")
        if isinstance(obj, Particle):
            self.sel_target_kind.set(obj.target_kind)
        else:
            self.sel_target_kind.set("nearest")
        if hasattr(obj, "w"):
            self.sel_w_var.set(obj.w)
            self.sel_h_var.set(obj.h)
            self.sel_w_entry.state(["!disabled"])
            self.sel_h_entry.state(["!disabled"])
        else:
            self.sel_w_var.set(0)
            self.sel_h_var.set(0)
            self.sel_w_entry.state(["disabled"])
            self.sel_h_entry.state(["disabled"])

    def object_at(self, px, py, prefer_particle=True):
        p = Vec(px, py)
        # particles first (small, drawn on top)
        for obj in reversed(self.particles):
            if obj.pos.dist(p) <= obj.radius + 4:
                return obj
        for objs in (self.safes, self.exits, self.walls):
            for obj in reversed(objs):
                if obj.contains_point(p):
                    return obj
        return None

    def handle_at(self, rect, px, py, tol=7.0):
        """Return 'move' or a resize handle name for a point near a rect."""
        # corners
        corners = {"nw": (rect.left, rect.top), "ne": (rect.right, rect.top),
                   "sw": (rect.left, rect.bottom), "se": (rect.right, rect.bottom)}
        for name, (cx, cy) in corners.items():
            if math.hypot(px - cx, py - cy) <= tol:
                return name
        # edges
        if abs(px - rect.left) <= tol and rect.top - tol <= py <= rect.bottom + tol:
            return "w"
        if abs(px - rect.right) <= tol and rect.top - tol <= py <= rect.bottom + tol:
            return "e"
        if abs(py - rect.top) <= tol and rect.left - tol <= px <= rect.right + tol:
            return "n"
        if abs(py - rect.bottom) <= tol and rect.left - tol <= px <= rect.right + tol:
            return "s"
        if rect.contains_point(Vec(px, py)):
            return "move"
        return None

    # ------------------------------------------------------------- events --
    def on_click(self, event):
        if self.mode == "SIM":
            return
        obj = self.object_at(event.x, event.y)
        if self.tool == "select":
            self.selected = obj
            self.drag = {"obj": obj, "start": Vec(event.x, event.y)}
            if obj is not None and hasattr(obj, "w") and not isinstance(obj, Particle):
                h = self.handle_at(obj, event.x, event.y)
                self.drag["handle"] = h if h else "move"
            else:
                self.drag["handle"] = "move"
            self.update_selection_panel()
            self.status("Selected: " + (type(obj).__name__ if obj else "nothing"))
        elif self.tool == "wall":
            self.drag = {"obj": None, "start": Vec(event.x, event.y), "kind": "wall"}
        elif self.tool == "exit":
            self.drag = {"obj": None, "start": Vec(event.x, event.y), "kind": "exit"}
        elif self.tool == "safe":
            self.drag = {"obj": None, "start": Vec(event.x, event.y), "kind": "safe"}
        elif self.tool == "particle":
            p = Particle(event.x, event.y)
            self.particles.append(p)
            self.selected = p
            self.stats_particles.set(f"Particles: {len(self.particles)}")
            self.update_selection_panel()
            self.status("Particle placed.")
            self.redraw()
        elif self.tool == "delete":
            if obj is not None:
                lst = {"wall": self.walls, "exit": self.exits, "safe": self.safes,
                       "particle": self.particles}.get(obj.kind)
                if lst is not None and obj in lst:
                    lst.remove(obj)
                if self.engine and obj in self.engine.particles:
                    self.engine.particles.remove(obj)
                if self.selected is obj:
                    self.selected = None
                self.stats_particles.set(f"Particles: {len(self.particles)}")
                self.update_selection_panel()
                self.status("Deleted.")
                self.redraw()
        elif self.tool == "target":
            if isinstance(obj, Particle):
                self.pending_target = obj
                self.status("Target tool: now click an exit or safe zone.")
            elif self.pending_target is not None and obj in (self.exits + self.safes):
                self.pending_target.target_kind = "exit" if obj in self.exits else "safe"
                self.pending_target.target_obj = obj
                self.status(f"Particle targeted to {'exit' if obj in self.exits else 'safe zone'}.")
                self.pending_target = None
                self.redraw()
        self.redraw()

    def on_drag(self, event):
        if self.mode == "SIM" or self.drag is None:
            return
        cur = Vec(event.x, event.y)
        d = self.drag
        if d["kind"] in ("wall", "exit", "safe"):
            x0, y0 = d["start"].x, d["start"].y
            x1, y1 = cur.x, cur.y
            w = abs(x1 - x0); h = abs(y1 - y0)
            if w < MIN_RECT: w = MIN_RECT
            if h < MIN_RECT: h = MIN_RECT
            rx = min(x0, x1); ry = min(y0, y1)
            self.canvas.delete("dragprev")
            fill = {"wall": "#444", "exit": "#6f6", "safe": "#6cf"}[d["kind"]]
            self.canvas.create_rectangle(rx, ry, rx + w, ry + h, outline="black", width=2,
                                         fill=fill, tags="dragprev")
            return
        obj = d["obj"]
        if obj is None:
            return
        handle = d.get("handle", "move")
        dx = cur.x - d["start"].x
        dy = cur.y - d["start"].y
        if handle == "move":
            obj.x += dx; obj.y += dy
            d["start"] = cur
        elif hasattr(obj, "w") and not isinstance(obj, Particle):
            r = obj
            if handle == "n":
                nh = max(MIN_RECT, r.h - dy)
                r.y = r.bottom - nh; r.h = nh
            elif handle == "s":
                r.h = max(MIN_RECT, r.h + dy)
            elif handle == "w":
                nw = max(MIN_RECT, r.w - dx)
                r.x = r.right - nw; r.w = nw
            elif handle == "e":
                r.w = max(MIN_RECT, r.w + dx)
            elif handle == "nw":
                nh = max(MIN_RECT, r.h - dy); nw = max(MIN_RECT, r.w - dx)
                r.y = r.bottom - nh; r.h = nh; r.x = r.right - nw; r.w = nw
            elif handle == "ne":
                nh = max(MIN_RECT, r.h - dy); nw = max(MIN_RECT, r.w + dx)
                r.y = r.bottom - nh; r.h = nh; r.w = nw
            elif handle == "sw":
                nh = max(MIN_RECT, r.h + dy); nw = max(MIN_RECT, r.w - dx)
                r.x = r.right - nw; r.w = nw; r.h = nh
            elif handle == "se":
                r.w = max(MIN_RECT, r.w + dx); r.h = max(MIN_RECT, r.h + dy)
            d["start"] = cur
        self.update_selection_panel()
        self.redraw()

    def on_release(self, event):
        if self.mode == "SIM" or self.drag is None:
            return
        d = self.drag
        if d["kind"] in ("wall", "exit", "safe"):
            cur = Vec(event.x, event.y)
            w = abs(cur.x - d["start"].x); h = abs(cur.y - d["start"].y)
            self.canvas.delete("dragprev")
            if w >= MIN_RECT and h >= MIN_RECT:
                r = Rect(min(d["start"].x, cur.x), min(d["start"].y, cur.y), w, h)
                if d["kind"] == "wall":
                    self.walls.append(r)
                elif d["kind"] == "exit":
                    self.exits.append(r)
                else:
                    self.safes.append(r)
                self.status(f"{d['kind'].capitalize()} drawn.")
            self.drag = None
            self.redraw()
            return
        self.drag = None
        self.redraw()

    def on_right_click(self, event):
        if self.mode == "SIM":
            return
        obj = self.object_at(event.x, event.y)
        if obj is not None:
            lst = {"wall": self.walls, "exit": self.exits, "safe": self.safes,
                   "particle": self.particles}.get(obj.kind)
            if lst is not None and obj in lst:
                lst.remove(obj)
            if self.engine and obj in self.engine.particles:
                self.engine.particles.remove(obj)
            if self.selected is obj:
                self.selected = None
            self.stats_particles.set(f"Particles: {len(self.particles)}")
            self.update_selection_panel()
            self.status("Deleted (right-click).")
            self.redraw()

    def on_motion(self, event):
        pass

    # ---------------------------------------------------------------- sim --
    def toggle_run(self):
        if self.mode != "SIM":
            self.set_sim_mode()
            return
        if not self.particles:
            self.status("No particles to simulate.")
            return
        if not self.exits and not self.safes:
            self.status("No exits or safe zones defined.")
            return
        if self.engine is None:
            self.engine = PSOEngine(self.walls, self.exits, self.safes, self.particles)
        else:
            self.engine.particles = self.particles
            self.engine.walls = self.walls
            self.engine.exits = self.exits
            self.engine.safes = self.safes
        self.sim_running = not self.sim_running
        self.btn_run.config(text="Pause" if self.sim_running else "Run")
        self.status("Simulation running." if self.sim_running else "Paused.")
        if self.sim_running:
            self.run_frame()

    def step_once(self):
        if self.mode != "SIM":
            self.set_sim_mode()
        if not self.particles:
            self.status("No particles.")
            return
        if not self.exits and not self.safes:
            self.status("No exits or safe zones defined.")
            return
        if self.engine is None:
            self.engine = PSOEngine(self.walls, self.exits, self.safes, self.particles)
        else:
            self.engine.particles = self.particles
            self.engine.walls = self.walls
            self.engine.exits = self.exits
            self.engine.safes = self.safes
        it = max(1, int(self.pso_iters.get()))
        for _ in range(it):
            self.engine.step(bounds=Vec(CANVAS_W, CANVAS_H))
        self.stats_iter.set(f"Iter: {self.engine.iteration}")
        evac = sum(1 for p in self.particles if p.evacuated)
        self.stats_evac.set(f"Evacuated: {evac}/{len(self.particles)}")
        self.status(f"Stepped {it} iteration(s).")
        self.redraw()
        if evac == len(self.particles):
            self.status("All particles evacuated!")

    def reset_sim(self):
        self.sim_running = False
        self.btn_run.config(text="Run")
        if self.engine:
            self.engine.iteration = 0
        for p in self.particles:
            p.evacuated = False
            p.vel = Vec(random.uniform(-1, 1), random.uniform(-1, 1))
            p.pbest_pos = p.pos.copy()
            p.pbest_cost = INF
            p.target_obj = None if p.target_kind == "nearest" else p.target_obj
        self.stats_iter.set("Iter: 0")
        self.stats_evac.set(f"Evacuated: 0/{len(self.particles)}")
        self.status("Simulation reset.")
        self.redraw()

    def run_frame(self):
        if not self.sim_running:
            return
        if not self.particles:
            self.sim_running = False
            self.btn_run.config(text="Run")
            return
        it = max(1, int(self.pso_iters.get()))
        for _ in range(it):
            self.engine.step(bounds=Vec(CANVAS_W, CANVAS_H))
        self.stats_iter.set(f"Iter: {self.engine.iteration}")
        evac = sum(1 for p in self.particles if p.evacuated)
        self.stats_evac.set(f"Evacuated: {evac}/{len(self.particles)}")
        self.redraw()
        if evac == len(self.particles):
            self.sim_running = False
            self.btn_run.config(text="Run")
            self.status("All particles evacuated!")
            return
        self.sim_job = self.after(30, self.run_frame)

    # ------------------------------------------------------------- drawing --
    def draw_rect(self, c, r, fill, outline, width=2):
        c.create_rectangle(r.left, r.top, r.right, r.bottom,
                           fill=fill, outline=outline, width=width)

    def group_best_positions(self):
        groups = {}
        for p in self.particles:
            if p.evacuated:
                continue
            groups.setdefault(p.group_key(self.exits, self.safes), []).append(p)
        out = {}
        for key, plist in groups.items():
            best = min(plist, key=lambda p: p.pbest_cost)
            out[key] = best.pbest_pos.copy()
        return out

    def redraw(self):
        c = self.canvas
        c.delete("all")
        # walls
        for w in self.walls:
            self.draw_rect(c, w, "#5a5a5a", "#111111", 2)
        # exits (green) - primary evacuation targets
        for e in self.exits:
            self.draw_rect(c, e, "#8f8", "#0a0", 2)
            c.create_text(e.center.x, e.center.y, text="EXIT", fill="#0a0",
                          font=("Segoe UI", 9, "bold"))
        # safe zones (blue) - alternate objectives
        for s in self.safes:
            self.draw_rect(c, s, "#8cf", "#05a", 2)
            c.create_text(s.center.x, s.center.y, text="SAFE", fill="#05a",
                          font=("Segoe UI", 9, "bold"))
        # group-best markers (where each target-group converges)
        if self.show_gbest.get():
            for pos in self.group_best_positions().values():
                c.create_oval(pos.x - 5, pos.y - 5, pos.x + 5, pos.y + 5,
                              fill="#ff0", outline="#a80", width=2)
                c.create_text(pos.x, pos.y - 11, text="g*", fill="#a80",
                              font=("Segoe UI", 8, "bold"))
        # particles (green orbs / people)
        for p in self.particles:
            col = "#4a4" if p.evacuated else "#2e8b2e"
            c.create_oval(p.pos.x - p.radius, p.pos.y - p.radius,
                          p.pos.x + p.radius, p.pos.y + p.radius,
                          fill=col, outline="#222", width=1,
                          tags=("particle", "selected" if p is self.selected else ""))
            if p is self.selected:
                c.create_oval(p.pos.x - p.radius - 4, p.pos.y - p.radius - 4,
                              p.pos.x + p.radius + 4, p.pos.y + p.radius + 4,
                              outline="#ff0", width=2)
            if self.show_vel.get() and not p.evacuated and p.vel.length() > 0.1:
                end = p.pos + p.vel * 3.0
                c.create_line(p.pos.x, p.pos.y, end.x, end.y, fill="#d00", width=2, arrow=tk.LAST)
                c.create_oval(end.x - 2, end.y - 2, end.x + 2, end.y + 2, fill="#d00")
            if self.show_pbest.get() and not p.evacuated:
                c.create_oval(p.pbest_pos.x - 2, p.pbest_pos.y - 2,
                              p.pbest_pos.x + 2, p.pbest_pos.y + 2,
                              fill="#0a0", outline="")
        # selection handles for editable rectangles
        obj = self.selected
        if (self.mode == "EDIT" and obj is not None and hasattr(obj, "w")
                and not isinstance(obj, Particle)):
            r = obj
            pts = {"nw": (r.left, r.top), "ne": (r.right, r.top),
                   "sw": (r.left, r.bottom), "se": (r.right, r.bottom),
                   "n": (r.center.x, r.top), "s": (r.center.x, r.bottom),
                   "w": (r.left, r.center.y), "e": (r.right, r.center.y)}
            for (cx, cy) in pts.values():
                c.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4,
                                   fill="#ff0", outline="#000")
        # empty-canvas hint
        if not (self.walls or self.exits or self.safes or self.particles):
            c.create_text(CANVAS_W / 2, CANVAS_H / 2, fill="#888",
                          text="Edit mode: use the left tools to build your map.\n"
                               "Draw walls/exits/safe zones by click-drag, place particles, then Simulate.",
                          justify="center", font=("Segoe UI", 11))

if __name__ == "__main__":
    app = App()
    app.mainloop()

