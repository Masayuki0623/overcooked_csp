import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
import threading
from collections import deque

# Try importing PIL for better image handling
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Try importing pygame for map rendering
try:
    import pygame
    from gym_cooking.misc.game.game import Game
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Pygame or Game class not found. Map preview disabled.")

try:
    try:
        from agent.agent.mind.llm_api import LLMService
    except ImportError:
        from agent.mind.llm_api import LLMService
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("LLMService not found. AI features disabled.")

class AgentConfigGUI:
    def __init__(self, env):
        self.env = env
        self.weights = {}
        self.text_input_value = ""
        self.constraint_input_value = ""
        self.generated_constraints = []
        self.forbidden_zones = []  # List of (x, y) tuples
        self.tasks = self._get_tasks_from_env()
        self.vars_dict = None
        self.image_cache = [] # To keep references to images alive
        self.selected_model = "gemini-2.5-flash"
        
        # Setup paths
        self.project_root = Path(os.getcwd())
        self.graphics_path = self.project_root / "testbed-cooking" / "gym_cooking" / "misc" / "game" / "graphics"
        self.prompt_path = self.project_root / "agent" / "prompts" / "weight_tuning" / "system_prompt.txt"
        self.constraint_prompt_path = self.project_root / "agent" / "prompts" / "constraint_generation" / "system_prompt.txt"
        
        self.root = tk.Tk()
        self.root.title("エージェント設定")
        self.center_window(self.root, 800, 650)
        
        # --- THEMING ---
        self.bg_color = "#FFE4B5" # Moccasin (Warm Kitchen)
        self.fg_color = "#5D4037" # Dark Brown
        self.accent_color = "#D7CCC8" # Light Brown/Beige
        
        # Font
        self.font_family = "Comic Sans MS"
        self.header_font = (self.font_family, 16, "bold")
        self.normal_font = (self.font_family, 11)
        
        self.root.configure(bg=self.bg_color)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=self.normal_font)
        self.style.configure("TButton", background=self.accent_color, foreground=self.fg_color, font=self.header_font)
        self.style.configure("TCombobox", fieldbackground="white", font=self.normal_font)
        self.style.configure("Vertical.TScrollbar", troughcolor=self.bg_color, background=self.accent_color)

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.current_frame = None
        self.show_main_menu()

    def _get_tasks_from_env(self):
        tasks = set()
        base_ingredients = ['lettuce', 'onion', 'tomato']
        
        for recipe in self.env.recipes:
            if hasattr(recipe, 'contents_names'):
                 contents = recipe.contents_names
            else:
                 contents = [c.name for c in recipe.contents if hasattr(c, 'name')]
            
            recipe_base_ings = []
            for c in contents:
                c_lower = c.lower()
                for base in base_ingredients:
                    if base in c_lower:
                        recipe_base_ings.append(base)
                        break
            
            if not recipe_base_ings:
                continue
            
            for ing in recipe_base_ings:
                tasks.add(f"chop_{ing}")
            
            unique_ings = sorted(list(set(recipe_base_ings)))
            recipe_name = getattr(recipe, 'full_name', '').lower()
            if not recipe_name:
                soup_name = '-'.join(unique_ings) + ' soup'
            else:
                csp_ings = [ing for ing in ['lettuce','onion','tomato'] if ing in recipe_name]
                if not csp_ings:
                    soup_name = '-'.join(unique_ings) + ' soup'
                else:
                    soup_name = '-'.join(csp_ings) + ' soup'

            tasks.add(f"cook_{soup_name}")
            tasks.add(f"serve_{soup_name}")
            
        return sorted(list(tasks))

    def center_window(self, window, width, height):
        window.update_idletasks()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = None

    def _save_current_values(self):
        if hasattr(self, 'text_var') and self.text_var:
            self.text_input_value = self.text_var.get()
        if hasattr(self, 'constraint_var') and self.constraint_var:
            self.constraint_input_value = self.constraint_var.get()
        if self.vars_dict:
            for t, var in self.vars_dict.items():
                self.weights[t] = var.get()
            self.vars_dict = None
            
    def _get_icon_for_task(self, task_name):
        filename = None
        target_size = (24, 24) # Smaller icons for compact view
        
        parts = task_name.split('_')
        verb = parts[0]
        obj = parts[1] if len(parts) > 1 else ""
        
        if verb == "chop":
            obj_cap = obj.capitalize()
            filename = f"Fresh{obj_cap}.png"
            if not (self.graphics_path / filename).exists():
                 filename = f"{obj_cap}.png"

        elif verb == "cook":
            ing_part = obj.replace(" soup", "")
            ings = [i.capitalize() for i in ing_part.split('-')]
            cooked_name = "-".join([f"Cooked{i}" for i in ings]) + ".png"
            
            if (self.graphics_path / cooked_name).exists():
                filename = cooked_name
            else:
                filename = "pot.png"
                
        elif verb == "serve":
            filename = "delivery.png"
        
        if not filename or not (self.graphics_path / filename).exists():
            return None
            
        filepath = self.graphics_path / filename
        
        try:
            if HAS_PIL:
                img = Image.open(filepath)
                img = img.resize(target_size, Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            else:
                photo = tk.PhotoImage(file=str(filepath))
                photo = photo.subsample(2, 2) 
            
            self.image_cache.append(photo) 
            return photo
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return None

    def _generate_map_image(self):
        if not HAS_PYGAME:
            return None
        try:
            # Create a temporary game instance for rendering
            # We assume env is fully initialized
            game = Game(self.env, play=False)
            game.on_init()
            game.on_render()
            
            # Convert pygame surface to PIL Image
            data = pygame.image.tostring(game.screen, 'RGB')
            w, h = game.screen.get_size()
            img = Image.frombytes('RGB', (w, h), data)
            
            game.on_cleanup()
            return img
        except Exception as e:
            print(f"Error generating map image: {e}")
            return None

    def strip_ansi(self, text):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', str(text))

    def _show_not_implemented(self):
        messagebox.showinfo("未実装", "この機能は将来のアップデートで追加される予定です。")

    def show_forbidden_area_config(self):
        self._save_current_values()
        
        # Clean up main_frame completely (remove scrollbar/canvas from main menu if any)
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.current_frame = None

        self.image_cache = []

        screen_height = self.root.winfo_screenheight()

        target_height = min(750, screen_height - 100) 
        self.center_window(self.root, 1100, target_height) # Widen window for side-by-side view

        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        header = ttk.Frame(self.current_frame)
        header.pack(fill=tk.X, pady=5)
        
        btn_back = tk.Button(header, text="⬅ 戻る", font=(self.font_family, 10, "bold"),
                             bg=self.accent_color, command=self.show_main_menu, relief="flat", padx=10)
        btn_back.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(header, text="ゾーン設定 (進入禁止エリア)", font=(self.font_family, 14, "bold")).pack(side=tk.LEFT, padx=20)
        
        # --- Tool Palette ---
        tool_frame = tk.Frame(self.current_frame, bg=self.bg_color, pady=5)
        tool_frame.pack(fill=tk.X, padx=20)
        
        self.tool_var = tk.StringVar(value="forbidden")
        
        lbl_tool = tk.Label(tool_frame, text="ペン選択:", bg=self.bg_color, font=self.normal_font)
        lbl_tool.pack(side=tk.LEFT)
        
        # Radio buttons for tools
        rb_forbidden = tk.Radiobutton(tool_frame, text="■ 進入禁止 (薄赤)", variable=self.tool_var, value="forbidden",
                                      bg=self.bg_color, fg="#D32F2F", selectcolor=self.bg_color, font=(self.font_family, 11, "bold"))
        rb_forbidden.pack(side=tk.LEFT, padx=15)
        
        rb_allowed = tk.Radiobutton(tool_frame, text="□ 進入可能 (白・解除)", variable=self.tool_var, value="allowed",
                                    bg=self.bg_color, fg="#333333", selectcolor=self.bg_color, font=(self.font_family, 11, "bold"))
        rb_allowed.pack(side=tk.LEFT, padx=15)
        
        tk.Label(tool_frame, text="※ドラッグで連続塗りつぶし可能", bg=self.bg_color, fg="#555", font=("MS Gothic", 9)).pack(side=tk.LEFT, padx=20)

        # Content has 2 columns: Map (Left) and Tasks (Right)
        # Use simple Frame with pack(side) instead of PanedWindow to ensure visibility
        split_frame = tk.Frame(self.current_frame, bg=self.bg_color)
        split_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left: Map container
        map_frame = tk.Frame(split_frame, bg=self.bg_color)
        map_frame.pack(side=tk.LEFT, anchor="nw") # Stuck to left, no expand

        # Right (but packed Left to be next to map): Tasks status
        task_frame = tk.LabelFrame(split_frame, text="タスク到達可能性", bg=self.bg_color, fg=self.fg_color, font=("MS Gothic", 10, "bold"), width=300)
        task_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 0), anchor="nw")
        task_frame.pack_propagate(False) # Enforce width
        
        # Canvas for Map (Aligned to Top-Left/NW)
        self.map_canvas = tk.Canvas(map_frame, bg="white", highlightthickness=1, highlightbackground="black")
        self.map_canvas.pack(anchor="nw", padx=0, pady=0) 

        # Task List UI
        self.task_labels = {}
        task_canvas = tk.Canvas(task_frame, bg=self.bg_color, highlightthickness=0)
        task_scrollbar = ttk.Scrollbar(task_frame, orient="vertical", command=task_canvas.yview)
        self.task_list_inner = tk.Frame(task_canvas, bg=self.bg_color)
        
        task_canvas.create_window((0, 0), window=self.task_list_inner, anchor="nw")
        task_canvas.configure(yscrollcommand=task_scrollbar.set)
        
        task_canvas.pack(side="left", fill="both", expand=True)
        task_scrollbar.pack(side="right", fill="y")
        
        self.task_list_inner.bind("<Configure>", lambda e: task_canvas.configure(scrollregion=task_canvas.bbox("all")))

        # Initialize Task List
        for task in self.tasks:
            lbl = tk.Label(self.task_list_inner, text=f"❓ {task}", bg=self.bg_color, fg="gray", font=("Consolas", 10), justify="left", anchor="w")
            lbl.pack(fill=tk.X, padx=5, pady=2)
            self.task_labels[task] = lbl

        # Get Map Dimensions
        try:
            world = self.env.world
            # 明示的に表示を更新して rep を生成させる
            if hasattr(world, 'update_display'):
                world.update_display()
            rows = world.height
            cols = world.width
            rep = world.rep 
        except Exception as e:
            world = None
            rows = 0
            cols = 0
            rep = []
            messagebox.showerror("エラー", f"マップ情報の取得に失敗しました: {e}")
            return

        if not rep or len(rep) != rows:
             messagebox.showerror("エラー", f"マップデータが無効です。")
             return

        # Calculate cell size
        max_canvas_w = 600
        max_canvas_h = 500
        
        if cols > 0 and rows > 0:
            cell_w = max_canvas_w // cols
            cell_h = max_canvas_h // rows
            self.cell_size = min(cell_w, cell_h, 60)
        else:
            self.cell_size = 30
        
        canvas_width = cols * self.cell_size
        canvas_height = rows * self.cell_size
        
        self.map_canvas.config(width=canvas_width, height=canvas_height)
        
        # Color mapping
        color_map = {
            ' ': 'white',   # Floor
            '-': '#D7CCC8', # Counter (color only if no image)
            '/': '#FFE082', # Cutboard
            '*': '#C5E1A5', # Delivery
            'U': '#B0BEC5', # Pot
            'T': '#FFAB91', # Tomato Src
            'L': '#A5D6A7', # Lettuce Src
            'O': '#FFE082', # Onion Src
            'P': '#EEEEEE', # Plate Src
            'B': '#BCAAA4'  # Bin
        }
        
        # Tile Image Mapping
        tile_files = {
            '/': 'cutboard.png',
            '*': 'delivery.png',
            'U': 'pot.png',
            'T': 'FreshTomatoTile.png',
            'L': 'FreshLettuceTile.png',
            'O': 'FreshOnionTile.png',
            'P': 'PlateTile.png',
            'B': 'bin.png'
        }

        # Preload and resize images
        self.tile_images = {} 
        self.overlay_image = None
        
        if HAS_PIL:
            # 1. Load tile images
            for char, filename in tile_files.items():
                filepath = self.graphics_path / filename
                if filepath.exists():
                    try:
                        pil_img = Image.open(filepath).convert("RGBA")
                        pil_img = pil_img.resize((self.cell_size, self.cell_size), Image.LANCZOS)
                        ph = ImageTk.PhotoImage(pil_img)
                        self.tile_images[char] = ph
                        self.image_cache.append(ph)
                    except Exception as e:
                        print(f"Failed to load image for {char}: {e}")
            
            # 2. Create overlay image (Red with alpha)
            try:
                # 80/255 alpha ~ 30% opacity
                overlay = Image.new('RGBA', (self.cell_size, self.cell_size), (255, 0, 0, 80)) 
                self.overlay_image = ImageTk.PhotoImage(overlay)
                self.image_cache.append(self.overlay_image)
            except Exception as e:
                print(f"Failed to create overlay: {e}")

        # Draw Grid
        self.overlay_ids = {} 
        
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        for r in range(rows):
            for c in range(cols):
                if c < len(rep[r]):
                    raw_char = rep[r][c]
                    char = ansi_escape.sub('', str(raw_char)).strip()
                    if not char: char = ' '
                    if len(char) > 1: char = char[0]
                else:
                    char = ' '

                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # 1. Base Layer (Background Color)
                base_color = color_map.get(char, 'white')
                self.map_canvas.create_rectangle(x1, y1, x2, y2, fill=base_color, outline="")

                # 2. Image Layer
                if char in self.tile_images:
                    self.map_canvas.create_image(x1 + self.cell_size//2, y1 + self.cell_size//2, image=self.tile_images[char])
                elif char != ' ' and char not in tile_files:
                    # If no image but not floor, maybe draw text as fallback (e.g. Counter)
                     if char == '-':
                         # Counter usually looks better with just color, but maybe add a line
                         self.map_canvas.create_line(x1, y2, x2, y2, fill="#8D6E63", width=2)
                     elif char not in [' ']:
                         self.map_canvas.create_text((x1+x2)/2, (y1+y2)/2, text=char, font=("Consolas", 10, "bold"), fill="#555")

                # 3. Forbidden Overlay Layer (Interactive)
                is_forbidden = (c, r) in self.forbidden_zones
                
                if self.overlay_image: 
                    state = 'normal' if is_forbidden else 'hidden'
                    ov_id = self.map_canvas.create_image(x1 + self.cell_size//2, y1 + self.cell_size//2, 
                                                         image=self.overlay_image, state=state)
                    self.overlay_ids[(r, c)] = {'type': 'image', 'id': ov_id}
                    
                    # Invisible rect for hit testing
                    self.map_canvas.create_rectangle(x1, y1, x2, y2, fill="", outline="#BDBDBD")
                else:
                    if is_forbidden:
                        fill_val = "red"
                        stipple_val = "gray50"
                    else:
                        fill_val = ""
                        stipple_val = ""
                    
                    rect_id = self.map_canvas.create_rectangle(x1, y1, x2, y2, 
                                                               fill=fill_val, 
                                                               stipple=stipple_val,
                                                               outline="#BDBDBD")
                    self.overlay_ids[(r, c)] = {'type': 'rect', 'id': rect_id}

        # Bind Click and Drag (Motion)
        self.map_canvas.bind("<Button-1>", lambda event: self._on_map_paint(event, cols, rows, rep))
        self.map_canvas.bind("<B1-Motion>", lambda event: self._on_map_paint(event, cols, rows, rep))
        
        # Initial validation
        self._validate_tasks(rows, cols, rep)

    def _validate_tasks(self, rows, cols, rep):
        # 0. Find Start Position (Assume first agent)
        start_pos = None
        for agent in self.env.sim_agents:
            start_pos = agent.location
            break
        
        if not start_pos:
            return

        # 1. BFS to find reachable cells (Flood Fill)
        # Avoid forbidden zones
        reachable = set()
        queue = deque([start_pos])
        visited = {start_pos}
        
        while queue:
            curr = queue.popleft()
            reachable.add(curr)
            
            cx, cy = curr
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    if (nx, ny) not in visited and (nx, ny) not in self.forbidden_zones:
                        # Wall check
                        if 0 <= ny < len(rep) and 0 <= nx < len(rep[ny]):
                            char = self.strip_ansi(rep[ny][nx]).strip()
                            # Treat empty space as walkable. 
                            # Note: In Overcooked, ' ' is floor. 
                            # Counters etc are obstacles. 
                            # However, we only care about standing locations.
                            # ' ' is usually the only walkable tile.
                            is_walkable = (char == '' or char == ' ')
                            
                            if is_walkable:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
        
        # 2. Check each task
        # Helper to check if any neighbor of target_char is reachable
        def is_reachable(target_chars, is_cook_task=False):
            found_target = False
            can_reach = False
            
            for r in range(rows):
                for c in range(cols):
                    if c < len(rep[r]):
                        char = self.strip_ansi(rep[r][c]).strip()
                        if char in target_chars:
                            found_target = True
                            # Check neighbors
                            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nx, ny = c + dx, r + dy
                                if (nx, ny) in reachable:
                                    can_reach = True
                                    break
                                
                                # Special exception for 'Cook' tasks:
                                # Allow standing in forbidden zone if it's adjacent to a reachable safe cell
                                # AND it's adjacent to the Pot.
                                if is_cook_task and (nx, ny) in self.forbidden_zones:
                                    # Check if this forbidden cell is reachable from a safe cell
                                    is_access_reachable = False
                                    for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                        nnx, nny = nx + ddx, ny + ddy
                                        if (nnx, nny) in reachable:
                                            is_access_reachable = True
                                            break
                                    if is_access_reachable:
                                        can_reach = True
                                        break
                    if can_reach: break
                if can_reach: break
            
            return found_target and can_reach

        for task, lbl in self.task_labels.items():
            possible = False
            
            if task.startswith("chop_"):
                # Needs Ingredient Source AND Cutboard
                ing_name = task.replace("chop_", "")
                # Map ingredient names to map chars
                # T: Tomato, L: Lettuce, O: Onion
                char_map = {'tomato': 'T', 'lettuce': 'L', 'onion': 'O'}
                target_char = char_map.get(ing_name)
                
                if target_char:
                    # Check Source Reachability
                    source_ok = is_reachable([target_char])
                    # Check Cutboard Reachability
                    cutboard_ok = is_reachable(['/'])
                    possible = source_ok and cutboard_ok
                else:
                    possible = False # Unknown ingredient
                    
            elif task.startswith("cook_"):
                # Needs Pot ('U')
                # Apply special exception
                possible = is_reachable(['U'], is_cook_task=True)
                
            elif task.startswith("serve_"):
                # Needs Delivery ('*')
                possible = is_reachable(['*'])
            
            # Update Label
            if possible:
                lbl.config(text=f"✅ {task}", fg="#2E7D32") # Green
            else:
                lbl.config(text=f"❌ {task}", fg="#C62828") # Red

    def _on_map_paint(self, event, cols, rows, rep):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        
        update_needed = False
        
        if 0 <= c < cols and 0 <= r < rows:
            mode = self.tool_var.get()
            is_forbidden = False
            
            if mode == "forbidden":
                if (c, r) not in self.forbidden_zones:
                    self.forbidden_zones.append((c, r))
                    is_forbidden = True
                    update_needed = True
                else:
                    is_forbidden = True
            else: # allowed
                if (c, r) in self.forbidden_zones:
                    self.forbidden_zones.remove((c, r))
                    is_forbidden = False
                    update_needed = True
                else:
                    return # Already allowed
            
            # Update Visual
            item = self.overlay_ids.get((r, c))
            if item:
                if item['type'] == 'image':
                     state = 'normal' if is_forbidden else 'hidden'
                     self.map_canvas.itemconfig(item['id'], state=state)
                else:
                    if is_forbidden:
                        self.map_canvas.itemconfig(item['id'], fill="red", stipple="gray50")
                    else:
                        self.map_canvas.itemconfig(item['id'], fill="", stipple="")
        
        if update_needed:
            self._validate_tasks(rows, cols, rep)

    def show_main_menu(self):
        self._save_current_values()
        # Clean up main_frame completely to rebuild with scrollbar
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        self.image_cache = []
        
        # --- Scrollable Container Setup ---
        canvas = tk.Canvas(self.main_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        self.current_frame = ttk.Frame(canvas) # This is the scrollable content area

        self.current_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.current_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- Content Construction (same as before but parented to self.current_frame) ---
        title_lbl = ttk.Label(self.current_frame, text="Overcooked エージェント設定", font=(self.font_family, 24, "bold"))
        title_lbl.pack(pady=(10, 5))

        # Top Section: Map (Left) and Buttons (Right)
        top_frame = tk.Frame(self.current_frame, bg=self.bg_color)
        top_frame.pack(fill=tk.X, padx=20, pady=5)

        # Left Panel for Map
        left_panel = tk.Frame(top_frame, bg=self.bg_color)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Panel for Buttons
        right_panel = tk.Frame(top_frame, bg=self.bg_color)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        # --- Map Preview (Left Panel) ---
        if HAS_PYGAME and HAS_PIL:
            map_img = self._generate_map_image()
            if map_img:
                # Resize if too big - Reduced to 180 as requested
                max_width = 180
                ratio = max_width / map_img.width
                new_height = int(map_img.height * ratio)
                map_img = map_img.resize((max_width, new_height), Image.LANCZOS)
                
                photo = ImageTk.PhotoImage(map_img)
                self.image_cache.append(photo)
                
                img_lbl = tk.Label(left_panel, image=photo, bg=self.bg_color, relief="solid", borderwidth=2)
                img_lbl.pack(pady=5)
                
                # Agent Locations
                agents_info = []
                for agent in self.env.sim_agents:
                    agents_info.append(f"{agent.name}: {agent.location}")
                
                loc_text = "Start Pos:\n" + "\n".join(agents_info) # Multiline for better fit
                tk.Label(left_panel, text=loc_text, bg=self.bg_color, fg="#333333", font=(self.font_family, 9), justify=tk.LEFT).pack(pady=2)

        # --- Buttons (Right Panel) ---
        # Smaller button font
        btn_font = (self.font_family, 11, "bold") # Reduced from 14
        
        # Priority Weights Button
        btn_w = tk.Button(right_panel, text="⚙ タスク優先度設定", 
                          font=btn_font, bg=self.accent_color, fg=self.fg_color,
                          command=self.show_weight_config, relief="flat", padx=10, pady=5)
        btn_w.pack(pady=5, fill=tk.X)

        # Future Setting 1
        btn_f1 = tk.Button(right_panel, text="🚫 進入禁止エリア設定", 
                          font=btn_font, bg=self.accent_color, fg=self.fg_color,
                          command=self.show_forbidden_area_config, relief="flat", padx=10, pady=5)
        btn_f1.pack(pady=5, fill=tk.X)

        # Future Setting 2
        btn_f2 = tk.Button(right_panel, text="⚙ 詳細設定 B (未実装)", 
                          font=btn_font, bg=self.accent_color, fg=self.fg_color,
                          command=self._show_not_implemented, relief="flat", padx=10, pady=5)
        btn_f2.pack(pady=5, fill=tk.X)
        
        # --- AI Section (Weights) ---
        ai_frame = tk.LabelFrame(self.current_frame, text=" AI 重み調整 ", bg=self.bg_color, fg=self.fg_color, font=self.header_font)
        ai_frame.pack(pady=5, padx=20, fill=tk.X)
        
        # Model Selection
        model_frame = tk.Frame(ai_frame, bg=self.bg_color)
        model_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(model_frame, text="モデル:", bg=self.bg_color, fg=self.fg_color, font=self.normal_font).pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.selected_model)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                   values=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview", "gemini-exp-1206", "gemini-pro-latest", "gemini-2.0-flash", "gemini-flash-latest"], 
                                   state="readonly", width=25, font=self.normal_font)
        model_combo.pack(side=tk.LEFT, padx=15)
        
        # Text Input
        tk.Label(ai_frame, text="指示 (例: '切る作業を優先して'):", bg=self.bg_color, fg=self.fg_color, font=self.normal_font).pack(anchor="w", padx=10)
        self.text_var = tk.StringVar(value=self.text_input_value)
        entry = tk.Entry(ai_frame, textvariable=self.text_var, font=self.normal_font)
        entry.pack(fill=tk.X, padx=10, pady=2, ipady=3)
        
        # Generate Button
        self.btn_gen = tk.Button(ai_frame, text="✨ AIで重みを生成", 
                            font=self.normal_font, bg="#81D4FA", fg=self.fg_color, # Light Blue
                            command=self.generate_with_ai, relief="flat", padx=10, pady=2)
        self.btn_gen.pack(pady=5)

        # --- AI Section (Constraints) ---
        cons_frame = tk.LabelFrame(self.current_frame, text=" AI 制約生成 ", bg=self.bg_color, fg=self.fg_color, font=self.header_font)
        cons_frame.pack(pady=5, padx=20, fill=tk.X)
        
        tk.Label(cons_frame, text="制約指示 (例: 'トマトを切る作業は同時に行わないで'):", bg=self.bg_color, fg=self.fg_color, font=self.normal_font).pack(anchor="w", padx=10)
        self.constraint_var = tk.StringVar(value=self.constraint_input_value)
        c_entry = tk.Entry(cons_frame, textvariable=self.constraint_var, font=self.normal_font)
        c_entry.pack(fill=tk.X, padx=10, pady=2, ipady=3)

        # Generate Constraints Button
        self.btn_gen_cons = tk.Button(cons_frame, text="✨ AIで制約を生成", 
                            font=self.normal_font, bg="#C5E1A5", fg=self.fg_color, # Light Green
                            command=self.generate_constraints_with_ai, relief="flat", padx=10, pady=2)
        self.btn_gen_cons.pack(pady=5)
        
        # Start Game Button
        btn_start_font = (self.font_family, 14, "bold")
        btn_start = tk.Button(self.current_frame, text="▶ ゲーム開始", 
                              font=btn_start_font, bg="#4CAF50", fg="white", # Green button
                              command=self.finish, relief="flat", padx=20, pady=10)
        btn_start.pack(pady=10, fill=tk.X, padx=50)

    def generate_with_ai(self):
        if not HAS_LLM:
            messagebox.showerror("エラー", "LLMService モジュールが見つかりません。")
            return

        instruction = self.text_var.get()
        if not instruction:
            messagebox.showwarning("警告", "AIへの指示を入力してください。")
            return
        
        self.selected_model = self.model_var.get()
        self.text_input_value = instruction # save
        
        original_text = self.btn_gen.cget("text")
        self.btn_gen.config(text="⏳ 思考中...", state="disabled")
        self.root.update()
        
        def run_inference():
            service = LLMService(model=self.selected_model)
            result = service.infer_weights(self.tasks, instruction, self.prompt_path)
            self.root.after(0, lambda: self._on_ai_complete(result, original_text))
            
        threading.Thread(target=run_inference, daemon=True).start()

    def _on_ai_complete(self, result, original_text):
        self.btn_gen.config(text=original_text, state="normal")
        
        if not result or "error" in result:
            err = result.get("error", "不明なエラー") if result else "応答なし"
            messagebox.showerror("AI エラー", f"重みの生成に失敗しました:\n{err}")
        else:
            updated_count = 0
            for task, weight in result.items():
                if task in self.tasks:
                    try:
                        w = float(weight)
                        self.weights[task] = w
                        updated_count += 1
                    except ValueError:
                        pass
            
            messagebox.showinfo("成功", f"AIの提案に基づいて {updated_count} 個のタスク優先度を更新しました。")
            self.show_weight_config()

    def generate_constraints_with_ai(self):
        if not HAS_LLM:
            messagebox.showerror("エラー", "LLMService モジュールが見つかりません。")
            return

        instruction = self.constraint_var.get()
        if not instruction:
            messagebox.showwarning("警告", "AIへの制約指示を入力してください。")
            return
        
        self.selected_model = self.model_var.get()
        self.constraint_input_value = instruction # save
        
        original_text = self.btn_gen_cons.cget("text")
        self.btn_gen_cons.config(text="⏳ 解析中...", state="disabled")
        self.root.update()
        
        def run_inference():
            service = LLMService(model=self.selected_model)
            result = service.infer_constraints(instruction, self.constraint_prompt_path)
            self.root.after(0, lambda: self._on_ai_constraint_complete(result, original_text))
            
        threading.Thread(target=run_inference, daemon=True).start()

    def _on_ai_constraint_complete(self, result, original_text):
        self.btn_gen_cons.config(text=original_text, state="normal")
        
        if not result or "error" in result:
            err = result.get("error", "不明なエラー") if result else "応答なし"
            messagebox.showerror("AI エラー", f"制約の解析に失敗しました:\n{err}")
        else:
            cons_list = result.get("constraints", [])
            if not cons_list:
                messagebox.showinfo("情報", "制約は生成されませんでした（指示が曖昧か、該当する制約タイプがありません）。")
                self.generated_constraints = []
            else:
                self.generated_constraints = cons_list
                # Show simple summary
                summary = "\n".join([f"- {c['type']}: {c.get('tasks', c.get('before', ''))}..." for c in cons_list])
                messagebox.showinfo("成功", f"以下の制約が生成されました:\n{summary}\n\nゲーム開始時に適用されます。")

    def show_weight_config(self):
        self._save_current_values()
        self.clear_frame()
        self.image_cache = [] 
        
        screen_height = self.root.winfo_screenheight()
        target_height = min(700, screen_height - 100)
        self.center_window(self.root, 800, target_height)
        
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        header = ttk.Frame(self.current_frame)
        header.pack(fill=tk.X, pady=5)
        
        btn_back = tk.Button(header, text="⬅ 戻る", font=(self.font_family, 10, "bold"),
                             bg=self.accent_color, command=self.show_main_menu, relief="flat", padx=10)
        btn_back.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(header, text="優先度調整", font=(self.font_family, 14, "bold")).pack(side=tk.LEFT, padx=20)
        
        content_frame = tk.Frame(self.current_frame, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        self.vars_dict = {}
        for t in self.tasks:
            # More compact row
            row = tk.Frame(content_frame, bg=self.bg_color, pady=2)
            row.pack(fill=tk.X, expand=False)
            
            icon = self._get_icon_for_task(t)
            if icon:
                icon_lbl = tk.Label(row, image=icon, bg=self.bg_color)
                icon_lbl.pack(side=tk.LEFT, padx=(0, 5))
            
            display_name = t.replace("_", " ").title()
            name_lbl = tk.Label(row, text=display_name, anchor="w", 
                                bg=self.bg_color, fg=self.fg_color, font=(self.font_family, 10))
            name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            slider_frame = tk.Frame(row, bg=self.bg_color)
            slider_frame.pack(side=tk.RIGHT)
            
            val = self.weights.get(t, 1.0)
            var = tk.DoubleVar(value=val)
            self.vars_dict[t] = var
            
            val_lbl = tk.Label(slider_frame, textvariable=var, width=4, bg=self.bg_color, fg=self.fg_color, font=(self.font_family, 10))
            val_lbl.pack(side=tk.RIGHT, padx=(5, 0))

            scale = tk.Scale(slider_frame, from_=0.1, to=10.0, resolution=0.1, variable=var, orient=tk.HORIZONTAL,
                             bg=self.bg_color, highlightthickness=0, fg=self.fg_color, length=150, sliderlength=15, width=10)
            scale.pack(side=tk.RIGHT)

    def finish(self):
        self._save_current_values()
        self.root.destroy()
        self.root.quit()

def configure_agent_settings(env):
    gui = AgentConfigGUI(env)
    gui.root.mainloop()
    return {
        'weights': gui.weights,
        'text_input': gui.text_input_value,
        'constraint_input': gui.constraint_input_value,
        'constraints': gui.generated_constraints,
        'forbidden_zones': gui.forbidden_zones
    }