import tkinter as tk
from tkinter import ttk
import os
from pathlib import Path

# Try importing PIL for better image handling
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class AgentConfigGUI:
    def __init__(self, env):
        self.env = env
        self.weights = {}
        self.text_input_value = ""
        self.tasks = self._get_tasks_from_env()
        self.vars_dict = None
        self.image_cache = [] # To keep references to images alive
        
        # Setup paths
        # Assuming current structure: agent/agent/myagent/gui.py
        # Project root is likely 3 levels up from here, or we use CWD if running from root
        # The user said CWD is C:\Users\sanda\PythonPrograms\OvercookedCspAgent
        self.project_root = Path(os.getcwd())
        self.graphics_path = self.project_root / "testbed-cooking" / "gym_cooking" / "misc" / "game" / "graphics"
        
        self.root = tk.Tk()
        self.root.title("Agent Configuration")
        self.center_window(self.root, 500, 600) # Slightly larger for icons
        
        # --- THEMING ---
        self.bg_color = "#FFE4B5" # Moccasin (Warm Kitchen)
        self.fg_color = "#5D4037" # Dark Brown
        self.accent_color = "#D7CCC8" # Light Brown/Beige
        
        # Font - Try to find a playful one
        self.font_family = "Comic Sans MS"
        self.header_font = (self.font_family, 16, "bold")
        self.normal_font = (self.font_family, 11)
        
        self.root.configure(bg=self.bg_color)
        
        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam' usually allows easier color customization
        
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=self.normal_font)
        self.style.configure("TButton", background=self.accent_color, foreground=self.fg_color, font=self.header_font)
        self.style.configure("TEntry", fieldbackground="white", font=self.normal_font)
        
        # Scrollbar style (simple tweak)
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
            
            # Normalize ingredients: extract base name if present
            recipe_base_ings = []
            for c in contents:
                c_lower = c.lower()
                for base in base_ingredients:
                    if base in c_lower:
                        recipe_base_ings.append(base)
                        break
            
            if not recipe_base_ings:
                continue
                
            # Chop tasks for each base ingredient
            for ing in recipe_base_ings:
                tasks.add(f"chop_{ing}")
            
            # Soup tasks
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
        if self.vars_dict:
            for t, var in self.vars_dict.items():
                self.weights[t] = var.get()
            self.vars_dict = None
            
    def _get_icon_for_task(self, task_name):
        """
        Loads an icon based on task name.
        Returns ImageTk.PhotoImage or tk.PhotoImage
        """
        filename = None
        target_size = (32, 32)
        
        parts = task_name.split('_')
        verb = parts[0]
        obj = parts[1] if len(parts) > 1 else ""
        
        if verb == "chop":
            # chop_lettuce -> FreshLettuce.png
            # Capitalize object: lettuce -> Lettuce
            obj_cap = obj.capitalize()
            filename = f"Fresh{obj_cap}.png"
            if not (self.graphics_path / filename).exists():
                 # Try finding just by containment if exact match fails
                 filename = f"{obj_cap}.png"

        elif verb == "cook":
            # cook_lettuce-onion soup -> try to construct CookedName
            # obj is like "tomato-onion soup"
            # Remove " soup" and capitalize parts
            ing_part = obj.replace(" soup", "")
            ings = [i.capitalize() for i in ing_part.split('-')]
            # Construct "CookedIng1-CookedIng2..."
            # Note: File names seem to be "CookedLettuce-CookedOnion.png"
            cooked_name = "-".join([f"Cooked{i}" for i in ings]) + ".png"
            
            if (self.graphics_path / cooked_name).exists():
                filename = cooked_name
            else:
                # Fallback to generic pot
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
                # Fallback to standard tk PhotoImage (PNG support depends on tk version)
                photo = tk.PhotoImage(file=str(filepath))
                # Simple subsample (only integers). 
                # Assuming original images are large (e.g. 64x64 or larger)
                # If 32x32 is target, we subsample by 2 or 3
                photo = photo.subsample(2, 2) 
            
            self.image_cache.append(photo) # Keep reference
            return photo
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return None

    def show_main_menu(self):
        self._save_current_values()
        self.clear_frame()
        
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        title_lbl = ttk.Label(self.current_frame, text="Overcooked Agent Setup", font=(self.font_family, 24, "bold"))
        title_lbl.pack(pady=(30, 20))
        
        # Priority Weights Button
        btn_font = (self.font_family, 14, "bold")
        
        btn_w = tk.Button(self.current_frame, text="⚙ Task Priorities", 
                          font=btn_font, bg=self.accent_color, fg=self.fg_color,
                          command=self.show_weight_config, relief="flat", padx=20, pady=10)
        btn_w.pack(pady=10, fill=tk.X, padx=50)
        
        # Text Input
        ttk.Label(self.current_frame, text="📝 Additional Instructions:").pack(pady=(20, 5), anchor="w", padx=50)
        self.text_var = tk.StringVar(value=self.text_input_value)
        entry = tk.Entry(self.current_frame, textvariable=self.text_var, font=self.normal_font)
        entry.pack(fill=tk.X, padx=50, ipady=5)
        
        # Start Game Button
        btn_start = tk.Button(self.current_frame, text="▶ START GAME", 
                              font=btn_font, bg="#4CAF50", fg="white", # Green button
                              command=self.finish, relief="flat", padx=20, pady=10)
        btn_start.pack(pady=40, fill=tk.X, padx=50)

    def show_weight_config(self):
        self._save_current_values()
        self.clear_frame()
        self.image_cache = [] 
        
        # Increase window size significantly to try fitting everything vertically
        # Getting screen size to max out height if needed
        screen_height = self.root.winfo_screenheight()
        target_height = min(900, screen_height - 100)
        self.center_window(self.root, 800, target_height)
        
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Frame(self.current_frame)
        header.pack(fill=tk.X, pady=10)
        
        btn_back = tk.Button(header, text="⬅ Back", font=(self.font_family, 12, "bold"),
                             bg=self.accent_color, command=self.show_main_menu, relief="flat")
        btn_back.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(header, text="Adjust Priorities", font=(self.font_family, 18, "bold")).pack(side=tk.LEFT, padx=20)
        
        # Main container
        # Use a canvas/scrollbar only if absolutely necessary, but user requested "no scrollbar if possible"
        # However, if items exceed screen height, they become inaccessible. 
        # I will use a frame that expands, but if it overflows, it overflows.
        # To be safe for "task name cut off", I will ensure wide horizontal layout.
        
        content_frame = tk.Frame(self.current_frame, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.vars_dict = {}
        for t in self.tasks:
            # Row container
            row = tk.Frame(content_frame, bg=self.bg_color, pady=5)
            row.pack(fill=tk.X, expand=False)
            
            # Icon
            icon = self._get_icon_for_task(t)
            if icon:
                icon_lbl = tk.Label(row, image=icon, bg=self.bg_color)
                icon_lbl.pack(side=tk.LEFT, padx=(0, 15))
            
            # Label - Use explicit width or weight to prevent overlap
            display_name = t.replace("_", " ").title()
            name_lbl = tk.Label(row, text=display_name, anchor="w", 
                                bg=self.bg_color, fg=self.fg_color, font=self.normal_font)
            # Push label to left, take available space up to slider
            name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Slider Container (Right side)
            slider_frame = tk.Frame(row, bg=self.bg_color)
            slider_frame.pack(side=tk.RIGHT)
            
            val = self.weights.get(t, 1.0)
            var = tk.DoubleVar(value=val)
            self.vars_dict[t] = var
            
            # Value Label
            val_lbl = tk.Label(slider_frame, textvariable=var, width=4, bg=self.bg_color, fg=self.fg_color, font=self.normal_font)
            val_lbl.pack(side=tk.RIGHT, padx=(10, 0))

            # Slider
            scale = tk.Scale(slider_frame, from_=0.1, to=10.0, resolution=0.1, variable=var, orient=tk.HORIZONTAL,
                             bg=self.bg_color, highlightthickness=0, fg=self.fg_color, length=200)
            scale.pack(side=tk.RIGHT)

    def finish(self):
        self._save_current_values()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return {
            'weights': self.weights,
            'text_input': self.text_input_value
        }

def configure_agent_settings(env):
    """
    GUIを起動して設定を取得する。
    戻り値: {'weights': dict, 'text_input': str}
    """
    gui = AgentConfigGUI(env)
    return gui.run()
