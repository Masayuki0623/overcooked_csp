import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
import threading

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
        self.tasks = self._get_tasks_from_env()
        self.vars_dict = None
        self.image_cache = [] # To keep references to images alive
        self.selected_model = "gemini-2.5-flash"
        
        # Setup paths
        self.project_root = Path(os.getcwd())
        self.graphics_path = self.project_root / "testbed-cooking" / "gym_cooking" / "misc" / "game" / "graphics"
        self.prompt_path = self.project_root / "agent" / "prompts" / "weight_tuning" / "system_prompt.txt"
        
        self.root = tk.Tk()
        self.root.title("エージェント設定")
        self.center_window(self.root, 800, 700)
        
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

    def _show_not_implemented(self):
        messagebox.showinfo("未実装", "この機能は将来のアップデートで追加される予定です。")

    def show_main_menu(self):
        self._save_current_values()
        self.clear_frame()
        self.image_cache = []
        
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
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
        btn_f1 = tk.Button(right_panel, text="⚙ 詳細設定 A (未実装)", 
                          font=btn_font, bg=self.accent_color, fg=self.fg_color,
                          command=self._show_not_implemented, relief="flat", padx=10, pady=5)
        btn_f1.pack(pady=5, fill=tk.X)

        # Future Setting 2
        btn_f2 = tk.Button(right_panel, text="⚙ 詳細設定 B (未実装)", 
                          font=btn_font, bg=self.accent_color, fg=self.fg_color,
                          command=self._show_not_implemented, relief="flat", padx=10, pady=5)
        btn_f2.pack(pady=5, fill=tk.X)
        
        # --- AI Section (Bottom) ---
        ai_frame = tk.LabelFrame(self.current_frame, text=" AI アシスタント ", bg=self.bg_color, fg=self.fg_color, font=self.header_font)
        ai_frame.pack(pady=10, padx=20, fill=tk.X)
        
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

    def show_weight_config(self):
        self._save_current_values()
        self.clear_frame()
        self.image_cache = [] 
        
        screen_height = self.root.winfo_screenheight()
        target_height = min(900, screen_height - 100)
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
        'text_input': gui.text_input_value
    }